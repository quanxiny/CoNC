"""Native static baseline evaluation of existing benchmark sources; never execute C."""
from __future__ import annotations
import argparse, gzip, json, sys, time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from . import campaign as c

sys.path.insert(0,str(c.ROOT/'external_eval'))
sys.path.insert(0,str(c.ROOT/'external_eval/baselines/deepwukong/deps'))
import static_baselines as native
import overflowgen_static_eval as old
import make_manifest as front

WORK=c.OUT/'static'


def prepare():
    c.identity();native.verify_tools();WORK.mkdir(parents=True,exist_ok=True)
    manifest=[r for r in c.csvread(c.PRIOR/'source_manifest.csv') if r['split']=='test']
    data=c.ROOT/'datasets/overflowgen_24455320/data/overflow_gen';raw_json={};by_case=defaultdict(list)
    queries=[]
    for r in manifest:
        _,split,index=r['sample_id'].split(':')
        if split not in raw_json:raw_json[split]=c.read(data/(split+'_data.json'))
        x=raw_json[split][index];assert int(x['label'])==int(r['label'])
        q={'sample_id':r['sample_id'],'source_row':int(r['row']),'label':int(r['label']),
           'group':r['group'],'source_case':int(r['source_case']),'query_line':int(x['line_num'][x['key_line']])}
        queries.append(q);by_case[q['source_case']].append(q)
    sources=[]
    for case,qs in sorted(by_case.items()):
        digits=f'{case:09d}';path=data/'source-code'/digits[:3]/digits[3:6]/digits[6:]/f'{case}.c'
        source=path.read_bytes();tree,functions=front.function_info(source,'.c')
        sanitized,names,reasons=front.neutralize(source,tree,functions)
        if tree.root_node.has_error:reasons.append('SOURCE_PARSE_ERROR')
        if [f['name'] for f in functions]!=['main']:reasons.append('ENTRY_STRUCTURE_MISMATCH')
        assert source.count(b'\n')==sanitized.count(b'\n')
        digest=front.sha(sanitized);target=WORK/'sources'/f'{digest}.c'
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert c.sha(target)==digest
        else:target.write_bytes(sanitized)
        sources.append({'scene_id':f'strict-source:{case}','source_case':case,'source_path':str(target),
                        'source_sha256':digest,'raw_path':str(path),'raw_sha256':c.sha(path),
                        'registered_queries':sorted({q['query_line'] for q in qs}),
                        'status':'UNKNOWN' if reasons else 'READY','failure_reason':';'.join(reasons)})
    c.save(WORK/'queries.json',queries);c.save(WORK/'sources.json',sources)
    protocol={'status':'FROZEN_BEFORE_NATIVE_ANALYSIS','targets':len(queries),'source_programs':len(sources),
              'queries':c.sha(WORK/'queries.json'),'sources':c.sha(WORK/'sources.json'),
              'native_runner':c.sha(native.__file__),'attribution_adapter':c.sha(old.__file__),
              'sanitizer':c.sha(front.__file__),'script':c.sha(__file__),
              'tool_registry':c.sha(native.WORK/'tool_registry.json'),'native_config':c.sha(native.CONFIG),
              'strict_freeze':c.sha(c.OUT/'freeze.json'),'source_program_execution':False,
              'query_policy':'Exact original target line; native primary location only; failures UNKNOWN; no label-directed remapping'}
    c.save(WORK/'protocol.json',protocol)
    print(json.dumps(protocol),flush=True)


def run(tool,workers):
    protocol=c.read(WORK/'protocol.json');native.verify_tools()
    for path,key in [(native.__file__,'native_runner'),(old.__file__,'attribution_adapter'),(__file__,'script'),
                     (native.CONFIG,'native_config'),(native.WORK/'tool_registry.json','tool_registry'),
                     (WORK/'queries.json','queries'),(WORK/'sources.json','sources')]:assert c.sha(path)==protocol[key]
    sources=c.read(WORK/'sources.json');queries=c.read(WORK/'queries.json');config=native.signed(native.CONFIG)
    identity={'strict_protocol_sha256':c.sha(WORK/'protocol.json'),'execution_script_sha256':c.sha(native.__file__),
              'config_sha256':c.sha(native.CONFIG),'tool_registry_sha256':c.sha(native.WORK/'tool_registry.json')}
    grouped=defaultdict(list)
    for r in queries:grouped[r['source_case']].append(r)
    folder=WORK/tool;folder.mkdir(parents=True,exist_ok=True);ledger=[];counts=Counter();start=time.monotonic()
    def one(source):
        assert c.sha(source['raw_path'])==source['raw_sha256']
        if source['status']!='READY':return source,{'status':'UNKNOWN','failure_reason':source['failure_reason'],'selected_alerts':[]},True
        receipt,cached=native.run_unit(tool,source,folder/'units'/str(source['source_case']),config,identity)
        return source,receipt,cached
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,s) for s in sources]
        for future in as_completed(futures):
            source,receipt,cached=future.result()
            pred,failure,hits,outside,unresolved=old.attribute(tool,receipt,source['source_path'],source['registered_queries'])
            for q in grouped[source['source_case']]:
                ledger.append({**q,'tool':tool,'status':'UNKNOWN' if failure else 'PASS','prediction':None if failure else pred[q['query_line']],
                               'failure_reason':failure or '', 'matched_alerts':hits.get(q['query_line'],0)})
            counts['completed']+=1;counts['unknown_sources']+=bool(failure);counts['cached']+=cached
            if counts['completed']%500==0:print(json.dumps({'tool':tool,**counts,'seconds':time.monotonic()-start}),flush=True)
    ledger.sort(key=lambda r:r['source_row']);assert len(ledger)==11769
    path=folder/'predictions.csv';c.t.write_csv(path,ledger)
    c.save(folder/'completion.json',{'status':'COMPLETE','protocol':c.sha(WORK/'protocol.json'),'sha256':c.sha(path),
            'targets':len(ledger),'coverage':dict(Counter(r['status'] for r in ledger)), 'elapsed_s':time.monotonic()-start,
            'source_status':dict(counts),'source_program_execution':False})
    print(json.dumps(c.read(folder/'completion.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run']);p.add_argument('--tool',choices=list(native.TOOLS));p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.phase=='prepare':prepare()
    else:run(a.tool,a.workers)

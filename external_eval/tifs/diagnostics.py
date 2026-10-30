"""Independent arithmetic oracle, real C -> pinned original frontend -> original tensors."""
import argparse,concurrent.futures,os,re,resource,signal,subprocess,sys,time
from collections import defaultdict,Counter
from .common import *

def template_hash(lines):
    names={};answer=[]
    for line in lines:
        seq=[]
        for token in line:
            if re.fullmatch(r'[+-]?\d+',token):token='<NUM>'
            elif re.fullmatch(r'(VAR|FUN)\d+',token):token=names.setdefault(token,'ID'+str(len(names)))
            seq.append(token)
        answer.append(seq)
    return digest(answer)

def bank():
    cfg=config()['diagnostics'];specs=[]
    for capacity in cfg['values']:
        for kind in ['index','copy']:
            for op in cfg['operations']:
                for depth in ([0] if op=='direct' else [d for d in cfg['depths'] if d]):
                    pair=digest([kind,capacity,op,depth])
                    for side in [0,1]:
                        value=capacity-1+side if kind=='index' else capacity+side
                        first=value-depth if op=='add' else value+depth if op=='sub' else value
                        expr=str(first)+(' + 1'*depth if op=='add' else ' - 1'*depth if op=='sub' else '')
                        # Oracle is construction arithmetic, never Builder's exact-value side data.
                        oracle_value=first+depth if op=='add' else first-depth if op=='sub' else first
                        assert oracle_value==value
                        label=int(not (0<=value<capacity) if kind=='index' else not (0<=value<=capacity))
                        declarations=[f'\tchar destination[{capacity}];']
                        if kind=='copy':declarations.append(f'\tchar source[{capacity+1}];')
                        body=declarations+[f'\tint count = {expr};', '\tdestination[count] = 0;' if kind=='index' else '\tmemcpy(destination, source, count);']
                        source=('#include <string.h>\n' if kind=='copy' else '')+'void fixture(void) {\n'+'\n'.join(body)+'\n}\n'
                        spec={'pair_id':pair,'label':label,'kind':kind,'operation':op,'depth':depth,'capacity':capacity,'value':value,
                              'literal_max':max(capacity+(kind=='copy'),first,1),'expression':expr,'source':source,
                              'group':kind+':'+op,'construction_variant':kind+':'+op+':'+str(depth),'oracle':'independent_nonnegative_int_char_array_bounds',
                              'oracle_value':oracle_value,'equivalence_id':digest([kind,capacity,value])}
                        spec['sample_id']='diag:'+digest(spec);specs.append(spec)
    # Guard flip pairs use the same actual index; only the necessary condition changes.
    for capacity in cfg['values']:
        pair=digest(['guard',capacity])
        for comparator,label in [('<',0),('<=',1)]:
            source=f'void fixture(void) {{\n\tchar destination[{capacity}];\n\tint count = {capacity};\n\tif (count {comparator} {capacity}) {{\n\t\tdestination[count] = 0;\n\t}}\n}}\n'
            spec={'pair_id':pair,'label':label,'kind':'guard','operation':'direct','depth':0,'capacity':capacity,'value':capacity,'literal_max':capacity,
                  'expression':str(capacity),'source':source,'group':'guard','oracle':'independent_constant_guard_then_char_index',
                  'oracle_value':capacity,'equivalence_id':digest(['guard',capacity,label])}
            spec['sample_id']='diag:'+digest(spec);specs.append(spec)
    for capacity in cfg['values']:
        for order in ['forward','swapped']:
            pair=digest(['operand_swap_add',capacity,order])
            for label in [0,1]:
                base=capacity-2+label;expr=f'{base} + 1' if order=='forward' else f'1 + {base}';value=base+1
                source=f'void fixture(void) {{\n\tchar destination[{capacity}];\n\tint count = {expr};\n\tdestination[count] = 0;\n}}\n'
                spec={'pair_id':pair,'label':label,'kind':'index','operation':'swap_add','depth':1,'capacity':capacity,'value':value,'literal_max':capacity,'expression':expr,'source':source,
                      'group':'index:add','construction_variant':'swap_add:'+order,'oracle':'independent_nonnegative_int_char_array_bounds','oracle_value':value,'equivalence_id':digest(['index',capacity,value])}
                spec['sample_id']='diag:'+digest(spec);specs.append(spec)
        pair=digest(['operand_swap_sub',capacity])
        for label in [0,1]:
            expr=f'{capacity+1} - 2' if label==0 else f'2 - {capacity+1}';value=capacity-1 if label==0 else 1-capacity
            source=f'void fixture(void) {{\n\tchar destination[{capacity}];\n\tint count = {expr};\n\tdestination[count] = 0;\n}}\n'
            spec={'pair_id':pair,'label':label,'kind':'index','operation':'swap_sub','depth':1,'capacity':capacity,'value':value,'literal_max':capacity+1,'expression':expr,'source':source,
                  'group':'index:sub','construction_variant':'swap_sub','oracle':'independent_signed_int_char_array_bounds','oracle_value':value,'equivalence_id':digest(['index',capacity,value])}
            spec['sample_id']='diag:'+digest(spec);specs.append(spec)
    bypair=defaultdict(list)
    for s in specs:bypair[s['pair_id']].append(s)
    selected=[]
    for pair in sorted(bypair,key=lambda p:digest([cfg['pair_seed'],p])):
        if len(selected)+len(bypair[pair])>cfg['max_programs']:break
        selected.extend(bypair[pair])
    return selected

def worker(spec_file,directory):
    imports();import legacy_frontend as legacy
    from mit_numeric_ood import encode_extended
    import numpy as np
    spec=read(spec_file);d=Path(directory);d.mkdir(parents=True,exist_ok=True)
    if (d/'result.json').exists():return read(d/'result.json')
    (d/'source').mkdir(exist_ok=True)
    source=d/'source/input.c'
    if source.exists() and source.read_text()!=spec['source']:raise ValueError('Diagnostic source changed')
    source.write_text(spec['source'])
    start=time.monotonic();result={'sample_id':spec['sample_id'],'label':spec['label'],'source_sha256':sha(source),'status':'UNKNOWN','stage':'literal_contract','slices':[]}
    try:
        if spec['literal_max']>32767 or re.search(r'= -',spec['source']):raise ValueError('NUMERIC_CONTRACT_UNSUPPORTED')
        api,jars,vendor=legacy.verify_assets(ROOT/'external_eval/vendor/legacy_joern')
        original,sources=legacy.load_original_functions(api)
        result['stage']='parser';table_root=d/'tables'
        command=[(__import__('shutil').which('java') or 'java'),'-Xmx512m','-cp',os.pathsep.join(map(str,jars)),'tools.parser.ParserMain','-outformat','csv','-outdir',str(table_root),str(source.parent)]
        if not table_root.exists():
            with (d/'parser.log').open('wb') as log:
                r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=config()['diagnostics']['parser_timeout_seconds'])
            if r.returncode:raise RuntimeError('PARSER_EXIT_'+str(r.returncode))
        table,_=legacy.csv_tables(table_root,source.name)
        result['stage']='slice_and_encode'
        # The sole approved gate extension; both ordinary and large literals use identical math.
        legacy.encode=encode_extended
        with legacy.preprocessing_deadline(config()['diagnostics']['preprocess_timeout_seconds']):
            parsed=legacy.public_slices(source,table,original,api)
        arrays=[]
        for i,item in enumerate(parsed['slices']):
            if item['status']!='ENCODED':raise ValueError('REQUIRED_SLICE_REJECTED:'+item.get('reason','unknown'))
            x,g=encode_extended(item['public'],parsed['line_edges'],original)
            arrays.append(x)
            result['slices'].append({'sample_id':spec['sample_id']+':slice'+str(i),'public':item['public'],'template_hash':template_hash(item['public']['line_code']),
                                     'tensor_cache':str(d/'tensors.npz'),'tensor_row':i,'input_sha256':canonical_input(x),
                                     'key_source_line':item['key_source_line'],'node_count':int(x['node_count'])})
        if not arrays:raise ValueError('NO_CANDIDATE_SLICES')
        # The oracle location is checked after all automatic candidates are extracted.
        sink_line=next(i+1 for i,line in enumerate(spec['source'].splitlines()) if 'destination[count]' in line or 'memcpy(' in line)
        if not any(r['key_source_line']==sink_line for r in result['slices']):raise ValueError('TARGET_ACCESS_NOT_EXTRACTED')
        np.savez(d/'tensors.npz',**{k:np.stack([x[k] for x in arrays]) for k in KEYS})
        result.update(status='PASS',stage='complete',tensor_sha256=sha(d/'tensors.npz'),vendor=vendor,
                      parser_tables_sha256={n:sha(table/n) for n in ['nodes.csv','edges.csv']})
    except Exception as error:
        result.update(reason=type(error).__name__+':'+str(error))
    result['elapsed_s']=time.monotonic()-start
    atomic_json(d/'result.json',result,immutable=True);return result

def generate(out):
    d=out/'diagnostics';d.mkdir(parents=True,exist_ok=True);specs=bank()
    manifest={'status':'GENERATED_NOT_PREPARED','specs':specs,'config':config()['diagnostics'],'generator_sha256':sha(__file__),
              'programs':len(specs),'pairs':len({s['pair_id'] for s in specs}),'construction_groups':len({s['group'] for s in specs}),
              'construction_variants':len({s.get('construction_variant',s['group']) for s in specs}),
              'labels_not_from_builder':True,'unseen_template_status':'determined later by extracted templates vs source TRAIN; not assumed'}
    atomic_json(d/'manifest.json',manifest,immutable=True)
    for spec in specs:
        folder=d/'cases'/spec['sample_id'].split(':')[1];folder.mkdir(parents=True,exist_ok=True)
        atomic_json(folder/'spec.json',spec,immutable=True)
    return manifest

def prepare(out,limit=None):
    manifest=generate(out);specs=manifest['specs'][:limit] if limit else manifest['specs']
    def one(spec):
        d=out/'diagnostics/cases'/spec['sample_id'].split(':')[1]
        if (d/'result.json').exists():
            result=read(d/'result.json')
            if result['status']=='PASS' and sha(d/'tensors.npz')!=result['tensor_sha256']:raise ValueError('Diagnostic cache changed')
            return result
        argv=[config()['runtime']['python'],'-m','external_eval.tifs.diagnostics','worker','--spec',str(d/'spec.json'),'--directory',str(d)]
        with (d/'worker.log').open('a') as log:
            p=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=config()['diagnostics']['parser_timeout_seconds']+40)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                try:code=p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
                if not (d/'result.json').exists():atomic_json(d/'result.json',{'sample_id':spec['sample_id'],'status':'UNKNOWN','stage':'worker','reason':'INDEPENDENT_WALL_TIMEOUT','slices':[]},immutable=True)
        if not (d/'result.json').exists():atomic_json(d/'result.json',{'sample_id':spec['sample_id'],'status':'UNKNOWN','stage':'worker','reason':'WORKER_EXIT_'+str(code),'slices':[]},immutable=True)
        return read(d/'result.json')
    with concurrent.futures.ThreadPoolExecutor(max_workers=config()['diagnostics']['max_workers']) as pool:
        for i,result in enumerate(pool.map(one,specs)):
            if i%20==0:print(json.dumps({'stage':'diagnostic_preparation','completed':i+1,'target':len(specs),'last_status':result['status']}),flush=True)
    ledger=[]
    with (out/'source_manifest.csv').open() as f:train_rows=[r for r in csv.DictReader(f) if r['split']=='train']
    train_templates={r['normalized_slice_sha256'] for r in train_rows};train_inputs={r['input_sha256'] for r in train_rows}
    for spec in manifest['specs']:
        d=out/'diagnostics/cases'/spec['sample_id'].split(':')[1]
        result=read(d/'result.json') if (d/'result.json').exists() else {'status':'NOT_PROCESSED','stage':'queued','slices':[]}
        seen=bool(result['slices']) and all(s['template_hash'] in train_templates for s in result['slices'])
        v=spec['literal_max'];numeric='0_999' if v<=999 else '1000_1023' if v<=1023 else '1024_32767'
        ledger.append({k:v for k,v in spec.items() if k!='source'}|{'status':result['status'],'failure_stage':result['stage'],'failure_reason':result.get('reason',''),
                      'known_exact_TRAIN_slice_overlap':any(s['input_sha256'] in train_inputs for s in result['slices']),
                      'numeric_stratum':numeric,'template_stratum':'seen_exact_normalized_slice_template' if seen else 'unseen_exact_normalized_slice_template',
                      'result_path':str(d/'result.json'),'result_sha256':sha(d/'result.json') if (d/'result.json').exists() else ''})
    write_csv(out/'diagnostics/ledger.csv',ledger)
    buckets=defaultdict(list)
    for r in ledger:
        if r['status']=='PASS':
            result=read(r['result_path']);buckets[digest(sorted(set(s['input_sha256'] for s in result['slices'])))].append(r)
    collisions=[]
    for h,items in buckets.items():
        pos=sum(r['label'] for r in items);neg=len(items)-pos
        if pos and neg:collisions.append({'input_set_sha256':h,'positive':pos,'negative':neg,'unavoidable_errors':min(pos,neg),'sample_ids':json.dumps([r['sample_id'] for r in items])})
    write_csv(out/'diagnostics/input_collisions.csv',collisions,['input_set_sha256','positive','negative','unavoidable_errors','sample_ids'])
    summary={'programs':len(ledger),'processed':sum(r['status']!='NOT_PROCESSED' for r in ledger),'pass':sum(r['status']=='PASS' for r in ledger),
             'groups':len({r['group'] for r in ledger}),'failures':dict(Counter(r['failure_reason'] for r in ledger if r['status']!='PASS')),
             'strata':dict(Counter(r['numeric_stratum']+':'+r['template_stratum']+':'+r['status'] for r in ledger)),
             'mixed_input_groups':len(collisions),'unavoidable_errors_lower_bound':sum(r['unavoidable_errors'] for r in collisions),
             'statistical_limit':'Seven broad construction families; many numeric variants are not independent templates. Narrow mechanism diagnostic, not large real-world generalization evidence.'}
    atomic_json(out/'diagnostics/summary.json',summary);return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['generate','prepare','worker']);p.add_argument('--out',type=Path,default=campaign_root());p.add_argument('--limit',type=int);p.add_argument('--spec');p.add_argument('--directory')
    a=p.parse_args()
    if a.mode=='worker':
        resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
        result=worker(a.spec,a.directory)
    else:result=generate(output_dir(a.out)) if a.mode=='generate' else prepare(output_dir(a.out),a.limit)
    print(json.dumps({k:v for k,v in result.items() if k not in ['specs','slices']}),flush=True)
if __name__=='__main__':main()

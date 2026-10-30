"""Refit native DeepDFA vocabulary on the new TRAIN; import its audited trainer."""
import argparse,importlib.util,sys,time
from collections import Counter,defaultdict
from types import SimpleNamespace
from .common import *

BASE=ROOT/'external_eval/baselines/deepdfa'
ASSETS=ROOT/'external_eval/paper_baselines/deepdfa'

def original():
    sys.path.insert(0,str(BASE))
    spec=importlib.util.spec_from_file_location('_tifs_original_deepdfa_training',BASE/'training.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def pack(out):
    frozen=require_freeze(out);t=original();d=out/'deepdfa/packed';d.mkdir(parents=True,exist_ok=True)
    binding={'freeze_sha256':sha(out/'freeze.json'),'training_source_sha256':sha(BASE/'training.py'),'source_manifest_sha256':sha(out/'source_manifest.csv')}
    atomic_json(d/'binding.json',binding,immutable=True)
    if (d/'packing.json').exists():return t.validate_packed(d)
    with (out/'source_manifest.csv').open() as f:rows={r['sample_id']:r for r in csv.DictReader(f)}
    graphs=[];coverage=[];inputs={};start=time.monotonic()
    prior=read(ASSETS/'packed/packing.json')['input_case_sha256']
    for i,(name,h) in enumerate(sorted(prior.items())):
        p=ROOT/name
        if sha(p)!=h:raise ValueError('Native feature cache changed: '+name)
        with gzip.open(p,'rt') as f:case=json.load(f)
        inputs[name]=h
        byid={g['sample_id']:g for g in case.get('features',{}).get('graphs',[])}
        for raw in case['record']['slices']:
            m=dict(raw);scope=case['record']['dataset'];g=byid.get(m['sample_id']);ok=bool(g and g['nodes'])
            if scope=='source':
                new=rows[m['sample_id']]
                if int(new['label'])!=m['label']:raise ValueError('DeepDFA label binding mismatch')
                m.update(split=new['split'],family_split=new['split'],group=new['group'])
            else:m['family_split']='external' # never fit vocabulary on target graphs
            m.update(dataset=scope,coverage=ok,coverage_status='accepted' if ok else case['status'])
            coverage.append(m)
            if ok:graphs.append({**g,**m,'family_split':m['split'] if scope=='source' else 'external'})
        if i%10000==0:print(json.dumps({'phase':'deepdfa_repack','cases':i,'elapsed_s':time.monotonic()-start}),flush=True)
    if {r['sample_id'] for r in coverage if r['dataset']=='source'}!=set(rows):raise ValueError('Incomplete new source manifest')
    vocabulary=t.fit(graphs);atomic_json(d/'vocabulary.json',vocabulary,immutable=True)
    scopes=defaultdict(list)
    for graph in graphs:scopes[graph['dataset']].append(t.encode(graph,vocabulary))
    for name,records in scopes.items():
        tmp=d/(name+'.pt.tmp');t.torch.save(records,tmp);tmp.replace(d/(name+'.pt'))
    with gzip.open(d/'coverage.jsonl.gz','wt') as f:
        for r in coverage:f.write(json.dumps(r,separators=(',',':'))+'\n')
    paths=[d/'vocabulary.json',d/'coverage.jsonl.gz']+[d/(name+'.pt') for name in scopes]
    result={'status':'COMPLETE','counts':{'total':len(coverage),'covered':len(graphs)},
            'source_counts':dict(Counter(r['split'] for r in coverage if r['dataset']=='source')),
            'input_case_sha256':inputs,'vocabulary_sha256':sha(d/'vocabulary.json'),'protocol_sha256':sha(ASSETS/'protocol.json'),
            'artifacts':{p.name:sha(p) for p in paths},'new_TRAIN_only_fit':True,'training_rerun_on_new_split':True}
    atomic_json(d/'packing.json',result,immutable=True);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pack','train','evaluate']);p.add_argument('--out',type=Path,default=campaign_root());p.add_argument('--seed',type=int,default=11)
    a=p.parse_args();out=output_dir(a.out);require_freeze(out)
    if a.mode=='pack':print(json.dumps(pack(out)['counts']));return
    t=original();args=SimpleNamespace(seed=a.seed,threads=8,data=str(out/'deepdfa/packed'),out=str(out/'deepdfa/runs'))
    if a.mode=='train':t.train(args)
    else:t.evaluate(args)
if __name__=='__main__':main()

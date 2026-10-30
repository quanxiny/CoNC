#!/usr/bin/env python3
"""Wait for the frozen frontend, pack, train3 seeds, then freeze external scores.

The classifier, feature fitting and DEV selection follow the frozen protocol.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];BASE=ROOT/'external_eval/paper_baselines/deepdfa'
PYTHON=__import__('sys').executable
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def run(command,log,env):
    print(json.dumps({'status':'LAUNCH','command':command,'log':str(log)}),flush=True)
    with log.open('a') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        result=process.wait()
    if result:raise RuntimeError('Stage failed; see '+str(log)+'; code='+str(result))
def main():
    p=argparse.ArgumentParser();p.add_argument('--frontend',default=str(BASE/'native_full'));p.add_argument('--frontend-run',default='full_v1');p.add_argument('--threads',type=int,default=8)
    p.add_argument('--controller-id',default='pipeline_cpu_ftz_v1');args=p.parse_args()
    out=BASE/'controllers'/args.controller_id;out.mkdir(parents=True,exist_ok=True)
    contract={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'training.py',HERE/'native_model.py',HERE/'pipeline.py',HERE/'evaluation.py',BASE/'protocol.json',BASE/'evaluation_contract.json',BASE/'cpu_runtime.json']}
    official=ROOT/'transformer_gru_v76_v2/third_party/DeepDFA/DDFA'
    for relative in ['sastvd/helpers/datasets.py','sastvd/scripts/dbize_absdf.py','sastvd/linevd/graphmogrifier.py','sastvd/linevd/dataset.py','configs/config_default.yaml','configs/config_bigvul.yaml','configs/config_ggnn.yaml']:
        path=official/relative;contract[str(path.relative_to(ROOT))]=sha(path)
    frozen=out/'contract.json'
    if frozen.exists():
        if json.loads(frozen.read_text())!=contract:raise ValueError('Controller code/protocol changed; preserve old experiment and inspect')
    else:put(frozen,contract)
    front=Path(args.frontend);receipt=front/'runs'/args.frontend_run/'receipt.json'
    put(out/'state.json',{'status':'WAITING_FRONTEND','pid':os.getpid(),'frontend_receipt':str(receipt)})
    print(json.dumps({'status':'WAITING_FRONTEND','pid':os.getpid(),'receipt':str(receipt)}),flush=True)
    while not receipt.exists():
        for relative,digest in contract.items():
            if sha(ROOT/relative)!=digest:raise ValueError('Frozen controller contract changed')
        time.sleep(20)
    frontend=json.loads(receipt.read_text())
    if frontend['counts'].get('batch_error'):raise ValueError('Frontend contains incomplete worker batches; resume frontend before training')
    # A source-level timeout remains an explicit uncovered record, never a safe label.
    import gzip
    cases=list(front.glob('cases/*/*.json.gz'))
    if len(cases)!=frontend['records']:raise ValueError('Frontend output case count incomplete')
    for case in cases:
        with gzip.open(case,'rt') as f:x=json.load(f)
        if x['source_manifest_sha256']!=frontend['manifest_sha256'] or x.get('retryable'):raise ValueError('Stale/incomplete frontend case')
    env=os.environ.copy();env.update(OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS='1')
    packed=BASE/'packed';packing=packed/'packing.json'
    if not packing.exists():
        put(out/'state.json',{'status':'PACKING','frontend_sha256':sha(receipt)})
        run([PYTHON,str(HERE/'training.py'),'pack','--frontend',str(front),'--data',str(packed)],out/'pack.log',env)
    info=json.loads(packing.read_text())
    if info['protocol_sha256']!=sha(BASE/'protocol.json'):raise ValueError('Packed protocol mismatch')
    for name,digest in info['artifacts'].items():
        if sha(packed/name)!=digest:raise ValueError('Packed data changed')
    for seed in [11,29,47]:
        put(out/'state.json',{'status':'TRAINING','seed':seed})
        run([PYTHON,str(HERE/'training.py'),'train','--seed',str(seed),'--threads',str(args.threads)],out/f'train_seed{seed}.log',env)
    for seed in [11,29,47]:
        put(out/'state.json',{'status':'FROZEN_EVALUATION','seed':seed})
        run([PYTHON,str(HERE/'training.py'),'evaluate','--seed',str(seed),'--threads',str(args.threads)],out/f'evaluate_seed{seed}.log',env)
    outputs={}
    for seed in [11,29,47]:
        run_dir=BASE/'runs'/f'seed{seed}'
        for pattern in ['completed.json','*_evaluation.json','*_predictions.jsonl.gz','*_predictions.sqlite']:
            for artifact in run_dir.glob(pattern):outputs[str(artifact.relative_to(ROOT))]=sha(artifact)
    result={'status':'COMPLETE','seeds':[11,29,47],'frontend_receipt_sha256':sha(receipt),'packing_sha256':sha(packing),'contract':contract,'outputs':outputs,
            'method':'official DeepDFA GGNN; matched-slice native CFG with whole-source feature/type context','checkpoint':'minimum full covered source DEV BCE','threshold':0.5,
            'all_three_seed_checkpoints_frozen_before_external_scoring':True}
    put(out/'completed.json',result);put(out/'state.json',{'status':'COMPLETE','completed_sha256':sha(out/'completed.json')});print(json.dumps({'status':'COMPLETE','path':str(out/'completed.json')}),flush=True)
if __name__=='__main__':main()

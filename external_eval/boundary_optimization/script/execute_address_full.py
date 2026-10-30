"""Thin full-source controller; same frozen one-query worker, source CPP reuse."""
from pathlib import Path
import sys,os,json,time,hashlib,threading,signal
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from external_eval.boundary_optimization.common import read,sha,atomic_json,write_csv
import execute_address_diagnostic as worker
from execute_binding_pilot import descendants,pid_stamp
B=ROOT/'external_eval/tifs_runs/boundary_binding_20261021';D=B/'address_full_4a768be154f1'
def certificate(a,folder,c):
    prior=c['request'].get('library_binding_reference')
    if prior:return prior
    p=folder/'library_binding/result.json'
    if not p.exists():return None
    r=read(p);b=r.get('binding')
    if r.get('status')!='BOUND' or not b:return None
    assert b['source_sha256']==a['source_sha256'];assert sha(b['source_CPP_file'])==b['source_CPP_sha256'];assert sha(b['compiler'])==b['compiler_sha256']
    for n,h in b['headers'].items():assert sha(n)==h,n
    files=[p,folder/'library_binding/binding.json',folder/'library_binding/source.c',Path(b['source_CPP_file']),Path(b['source_CPP_file']).parent/'cpp_original.json']
    return {'result_path':str(p),'files':{str(n):sha(n) for n in files},'reuse_scope':'identical source-specific CPP/header/prototype facts; no address outcome sharing'}
def main():
    f=read(D/'freeze.json')
    for n,h in f['files'].items():assert sha(ROOT/n)==h,n
    for n,h in f['runtime_files'].items():assert sha(n)==h,n
    assert hashlib.sha256(json.dumps(worker.environment(),sort_keys=True).encode()).hexdigest()==f['worker_environment_sha256']
    units=read(D/'manifest.json')['units'];assert len(units)==11577;receipt=D/'execution_receipt.json'
    if receipt.exists():
        r=read(receipt);assert r['freeze_sha256']==sha(D/'freeze.json')
        for n,h in r['outputs'].items():assert sha(D/n)==h,n
        for a in r['units']:
            assert sha(a['completion'])==a['sha256'];c=read(a['completion']);worker.validate(Path(a['completion']).parent,c['request'])
        print('VERIFIED_CACHED_FULL_SOURCE_ADDRESS');return
    initial=read(D/'library_references.json');cpprefs=read(D/'cpp_references.json');registry=dict(initial);registry_lock=threading.Lock();locks={a['source_sha256']:threading.Lock() for a in units};started=time.monotonic();entries=[];summary=[]
    atomic_json(D/'controller_start.json',{'pid':os.getpid(),'stage':str(D),'programs':len(locks),'queries':len(units),'freeze_sha256':sha(D/'freeze.json'),'log':'controller.log','labels_or_model_predictions_loaded':False})
    print(json.dumps({'controller_pid':os.getpid(),'queries':len(units),'programs':len(locks)}),flush=True)
    def run(a):
        with locks[a['source_sha256']]:
            folder=D/'units'/hashlib.sha256(a['unit_id'].encode()).hexdigest()
            if (folder/'completion.json').exists():
                q=read(folder/'completion.json')['request'];ref=q.get('library_binding_reference');cpp=q.get('cpp_reference')
            else:
                with registry_lock:ref=registry.get(a['source_sha256'])
                cpp=cpprefs.get(a['source_sha256'])
            folder,c=worker.one(a,D,f,cpp,ref);new=certificate(a,folder,c)
            if new:
                with registry_lock:registry[a['source_sha256']]=new
            return folder,c
    def stop(signum,frame):
        for pid,stamp in reversed(list(descendants(os.getpid()).items())):
            if pid!=os.getpid() and pid_stamp(pid)==stamp:
                try:os.kill(pid,signal.SIGKILL)
                except ProcessLookupError:pass
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop)
    with ThreadPoolExecutor(max_workers=f['limits']['workers']) as pool:
        fs={pool.submit(run,a):a for a in units}
        for future in as_completed(fs):
            a=fs[future];folder,c=future.result();r=c['result'];entries.append({'unit_id':a['unit_id'],'completion':str(folder/'completion.json'),'sha256':sha(folder/'completion.json')});summary.append({'unit_id':a['unit_id'],'group':a['group'],'source_sha256':a['source_sha256'],'potential_verdict':r['potential_verdict'],'unknown_reasons':';'.join(r.get('unknown_reasons',[])),'failure':c['failure'],'exit_code':c['exit_code'],'CPP_created':r.get('CPP_created',False),'library_certificate_reused':bool(c['request'].get('library_binding_reference')),'elapsed_seconds':c['elapsed_seconds']})
            if len(entries)%64==0 or len(entries)==len(units):
                progress={'status':'RUNNING' if len(entries)<len(units) else 'COMPLETE','pid':os.getpid(),'done':len(entries),'total':len(units),'elapsed_seconds':time.monotonic()-started,'potential_witnesses':sum(a['potential_verdict']==1 for a in summary),'process_failures':sum(bool(a['failure']) for a in summary),'CPP_created':sum(a['CPP_created'] for a in summary),'source_certificates_available':len(registry)};atomic_json(D/'progress.json',progress);print(json.dumps(progress),flush=True)
    entries.sort(key=lambda a:a['unit_id']);summary.sort(key=lambda a:a['unit_id']);atomic_json(D/'unit_index.json',entries,immutable=True);write_csv(D/'summary.csv',summary);atomic_json(D/'source_library_registry.json',registry,immutable=True)
    atomic_json(receipt,{'status':'COMPLETE_FULL_SOURCE_D_POS','freeze_sha256':sha(D/'freeze.json'),'units':entries,'count':len(entries),'source_programs':len(locks),'wall_seconds':time.monotonic()-started,'potential_count':sum(a['potential_verdict']==1 for a in summary),'process_failures':sum(bool(a['failure']) for a in summary),'CPP_created':sum(a['CPP_created'] for a in summary),'initial_certificate_sources':len(initial),'final_certificate_sources':len(registry),'new_Eva_neural_training':0,'labels_or_model_predictions_loaded':False,'outputs':{n:sha(D/n) for n in ['unit_index.json','summary.csv','source_library_registry.json']}},immutable=True)
    print('FULL_SOURCE_ADDRESS_COMPLETE',flush=True)
if __name__=='__main__':main()

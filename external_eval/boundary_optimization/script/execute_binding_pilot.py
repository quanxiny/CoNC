"""Frozen identity-only <=20-program queue, 8 workers, per-unit hard deadline."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,time,signal
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from external_eval.boundary_optimization.common import read,sha,atomic_json,write_csv
D=ROOT/'external_eval/tifs_runs/boundary_binding_20261021';BASE=__import__('sys').executable;SCRIPT=Path(__file__).parent

def verify():
    f=read(D/'pilot_freeze.json')
    for n,h in f['files'].items():assert sha(ROOT/n)==h,n
    for n,h in f['external_runtime_files'].items():assert sha(n)==h,n
    assert f['environment_sha256']==hashlib.sha256(json.dumps(dict(os.environ),sort_keys=True).encode()).hexdigest(),'controller environment changed'
    return f

def pid_stamp(pid):
    try:return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
    except (FileNotFoundError,ProcessLookupError,IndexError):return None

def descendants(pid):
    todo=[pid];found={}
    while todo:
        p=todo.pop();stamp=pid_stamp(p)
        if stamp is not None:found[p]=stamp
        try:todo.extend(int(v) for v in Path('/proc',str(p),'task',str(p),'children').read_text().split())
        except (FileNotFoundError,ProcessLookupError):pass
    return found

def validate_complete(folder,expected):
    c=read(folder/'completion.json');assert c['request']==expected
    for n,h in c['artifacts'].items():assert sha(folder/n)==h,n
    native=folder/'native_identity/completion.json'
    if native.exists():
        r=read(native)
        for n,h in r['artifacts'].items():assert sha(native.parent/n)==h,n
        for n,h in r.get('external_dependencies',{}).items():assert sha(n)==h,n
    return c

def one_unit(a,freeze):
    folder=D/'pilot_units'/hashlib.sha256((a['dataset']+'|'+a['unit_id']).encode()).hexdigest();folder.mkdir(parents=True,exist_ok=True)
    q={k:a[k] for k in ['dataset','unit_id','source_path','source_sha256','entry','stratum']};q.update(freeze_sha256=sha(D/'pilot_freeze.json'),limits=freeze['limits'],classification='NOT_SCORED')
    if (folder/'completion.json').exists():return folder,validate_complete(folder,q)
    if (folder/'request.json').exists():assert read(folder/'request.json')==q
    else:atomic_json(folder/'request.json',q,immutable=True)
    assert sha(a['source_path'])==a['source_sha256'];start=time.monotonic();failure=None;code=None
    # A prior interrupted adapter attempt is preserved, not silently retried.
    partial=(folder/'native_identity').exists() and any((folder/'native_identity').iterdir()) and not (folder/'native_identity/completion.json').exists()
    if partial:failure='INTERRUPTED_NATIVE_ATTEMPT_REQUIRES_SEPARATE_CONTROLLED_RECOVERY'
    else:
        with (folder/'worker.stdout').open('w') as out,(folder/'worker.stderr').open('w') as err:
            p=subprocess.Popen([BASE,str(SCRIPT/'binding_pilot_worker.py'),str(folder/'request.json'),str(folder/'native_identity'),str(folder/'worker_result.json')],cwd=ROOT,stdout=out,stderr=err,start_new_session=True)
            known={};deadline=start+freeze['limits']['wall_seconds']
            while p.poll() is None:
                known.update(descendants(p.pid))
                if time.monotonic()>=deadline:
                    failure='UNIT_WALL_TIMEOUT'
                    for pid,stamp in reversed(list(known.items())):
                        if pid_stamp(pid)==stamp:
                            try:os.kill(pid,signal.SIGKILL)
                            except ProcessLookupError:pass
                    break
                time.sleep(.1)
            try:code=p.wait(timeout=3)
            except subprocess.TimeoutExpired:p.kill();code=p.wait();failure=failure or 'UNIT_TERMINATION_TIMEOUT'
        if code!=0:failure=failure or ('WORKER_EXIT_'+str(code))
    result=read(folder/'worker_result.json') if (folder/'worker_result.json').exists() and not failure else {'identity_gate':'UNKNOWN','classification_gate':'UNKNOWN','targets':[],'failure':failure,'facts_used_for_classification':False}
    failure=failure or result.get('failure');artifacts={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='completion.json'}
    for p in folder.rglob('completion.json'):
        if p!=folder/'completion.json':artifacts[str(p.relative_to(folder))]=sha(p)
    c={'request':q,'status':'COMPLETE' if not failure else 'UNKNOWN_FAILURE','failure':failure,'exit_code':code,'elapsed_seconds':time.monotonic()-start,'result':result,'artifacts':artifacts,'no_input_C_execution':True,'no_model_or_training':True};atomic_json(folder/'completion.json',c,immutable=True)
    return folder,validate_complete(folder,q)

def main():
    f=verify();manifest=read(D/'pilot/manifest.json');receipt=D/'pilot_execution_receipt.json';units=manifest['units'];assert len(units)<=20
    if receipt.exists():
        r=read(receipt);assert r['freeze_sha256']==sha(D/'pilot_freeze.json')
        for n,h in r['outputs'].items():assert sha(D/n)==h,n
        for a in r['units']:
            assert sha(a['completion'])==a['sha256'];c=read(a['completion']);validate_complete(Path(a['completion']).parent,c['request']);assert sha(c['request']['source_path'])==c['request']['source_sha256']
        print('VERIFIED_CACHED_BINDING_PILOT');return
    started=time.monotonic();entries=[];summary=[];targetrows=[]
    qlink={a['unit_id']:a for a in map(json.loads,open(D/'query_linkage/results.jsonl'))}
    with ThreadPoolExecutor(max_workers=f['limits']['workers']) as pool:
        jobs={pool.submit(one_unit,a,f):a for a in units}
        for future in as_completed(jobs):
            a=jobs[future];folder,c=future.result();r=c['result'];q=qlink[a['unit_id']];ts=r.get('targets',[]);bound=[t for t in ts if t.get('identity_state')=='BOUND_SINGLE_ACCESS'];qt=[t for t in ts if q['source_access_id'] is not None and t.get('source_access_id')==q['source_access_id']]
            entries.append({'dataset':a['dataset'],'unit_id':a['unit_id'],'completion':str(folder/'completion.json'),'sha256':sha(folder/'completion.json')});summary.append({'dataset':a['dataset'],'unit_id':a['unit_id'],'stratum':a['stratum'],'source_query_linkage':q['linkage_status'],'identity_gate':r['identity_gate'],'target_roles':len(ts),'bound_target_roles':len(bound),'query_bound':bool(len(qt)==1 and qt[0].get('identity_state')=='BOUND_SINGLE_ACCESS'),'classification_gate':r['classification_gate'],'failure':c['failure'],'elapsed_seconds':c['elapsed_seconds']})
            for t in ts:targetrows.append({'dataset':a['dataset'],'unit_id':a['unit_id'],'source_access_id':t.get('source_access_id'),'role':t.get('role'),'identity_state':t.get('identity_state'),'unknown_reasons':';'.join(t.get('unknown_reasons',[])),'native_probe_kinstr':t.get('native_probe_kinstr'),'classification':'UNKNOWN'})
            print(json.dumps({'done':len(entries),'total':len(units),'dataset':a['dataset'],'failure':c['failure'],'bound_roles':len(bound)}),flush=True)
    summary.sort(key=lambda a:(a['dataset'],a['unit_id']));entries.sort(key=lambda a:(a['dataset'],a['unit_id']));write_csv(D/'pilot_identity_summary.csv',summary);write_csv(D/'pilot_target_identity.csv',targetrows);atomic_json(D/'pilot_native_index.json',entries,immutable=True)
    atomic_json(receipt,{'status':'COMPLETE_IDENTITY_ONLY','freeze_sha256':sha(D/'pilot_freeze.json'),'units':entries,'registered_programs':len(units),'wall_seconds':time.monotonic()-started,'failures':sum(bool(a['failure']) for a in summary),'classification_scored':False,'new_neural_training':0,'outputs':{n:sha(D/n) for n in ['pilot_identity_summary.csv','pilot_target_identity.csv','pilot_native_index.json']}},immutable=True)
    print('BINDING_PILOT_COMPLETE')
if __name__=='__main__':main()

"""Per-unit isolated static address diagnostics; explicit stage manifest and freeze."""
from pathlib import Path
import sys,os,json,hashlib,subprocess,time,signal,threading,argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from external_eval.boundary_optimization.common import read,sha,atomic_json,write_csv
from execute_binding_pilot import descendants,pid_stamp
BASE=__import__('sys').executable;SCRIPT=Path(__file__).parent;B=ROOT/'external_eval/tifs_runs/boundary_binding_20261021'
def environment():return {k:v for k,v in os.environ.items() if k not in ['SHLVL','_']}
def validate(folder,q):
    c=read(folder/'completion.json');assert c['request']==q
    for n,h in c['artifacts'].items():assert sha(folder/n)==h,n
    br=folder/'library_binding/result.json'
    if br.exists():
        binding=read(br).get('binding')
        if binding:
            assert sha(binding['source_CPP_file'])==binding['source_CPP_sha256']
            assert sha(binding['compiler'])==binding['compiler_sha256']
            for p,h in binding['headers'].items():assert sha(p)==h,p
    if q.get('cpp_reference'):
        for n,h in q['cpp_reference']['files'].items():assert sha(n)==h,n
    if q.get('library_binding_reference'):
        for n,h in q['library_binding_reference']['files'].items():assert sha(n)==h,n
        b=read(q['library_binding_reference']['result_path'])['binding']
        assert sha(b['source_CPP_file'])==b['source_CPP_sha256'] and sha(b['compiler'])==b['compiler_sha256']
        for n,h in b['headers'].items():assert sha(n)==h,n
    return c

def one(a,stage,f,cpp_reference=None,library_reference=None):
    folder=stage/'units'/hashlib.sha256(a['unit_id'].encode()).hexdigest();folder.mkdir(parents=True,exist_ok=True);q={k:a[k] for k in ['unit_id','source_path','source_sha256','entry','query_registration']};q.update(stage_freeze_sha256=sha(stage/'freeze.json'),limits=f['limits'],expression_budget=f['expression_budget'],cpp_reference=cpp_reference,library_binding_reference=library_reference)
    if (folder/'completion.json').exists():return folder,validate(folder,q)
    if (folder/'request.json').exists():assert read(folder/'request.json')==q
    else:atomic_json(folder/'request.json',q,immutable=True)
    started=time.monotonic();failure=None;exitcode=None
    if any(p.name!='request.json' for p in folder.iterdir()):failure='INTERRUPTED_UNIT_PRESERVED_REQUIRES_CONTROLLED_RECOVERY'
    else:
        with (folder/'stdout.txt').open('w') as out,(folder/'stderr.txt').open('w') as err:
            p=subprocess.Popen([BASE,str(SCRIPT/'address_diagnostic_worker.py'),str(folder/'request.json'),str(folder)],cwd=ROOT,env=environment(),stdout=out,stderr=err,start_new_session=True);deadline=started+q['limits']['wall_seconds'];known={}
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
            try:exitcode=p.wait(timeout=3)
            except subprocess.TimeoutExpired:p.kill();exitcode=p.wait();failure=failure or 'UNIT_TERMINATION_TIMEOUT'
        if exitcode!=0:failure=failure or ('WORKER_SIGNAL_'+str(-exitcode) if exitcode is not None and exitcode<0 else 'WORKER_EXIT_'+str(exitcode))
    result=read(folder/'result.json') if (folder/'result.json').exists() and not failure else {'unit_id':a['unit_id'],'potential_verdict':'UNKNOWN','unknown_reasons':[failure],'new_Eva_runs':0,'new_neural_training':0}
    assert result['potential_verdict'] in [1,'UNKNOWN'];artifacts={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p!=folder/'completion.json'};c={'request':q,'result':result,'failure':failure,'exit_code':exitcode,'elapsed_seconds':time.monotonic()-started,'artifacts':artifacts,'no_input_C_execution':True};atomic_json(folder/'completion.json',c,immutable=True);return folder,validate(folder,q)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);ap.add_argument('--preflight-only',action='store_true');args=ap.parse_args();stage=ROOT/args.stage;f=read(stage/'freeze.json')
    for n,h in f['files'].items():assert sha(ROOT/n)==h,n
    for n,h in f['runtime_files'].items():assert sha(n)==h,n
    assert f['worker_environment_sha256']==hashlib.sha256(json.dumps(environment(),sort_keys=True).encode()).hexdigest(),'scientific worker environment changed'
    units=read(stage/'manifest.json')['units'];receipt=stage/'execution_receipt.json'
    if receipt.exists():
        r=read(receipt);assert r['freeze_sha256']==sha(stage/'freeze.json')
        for n,h in r['outputs'].items():assert sha(stage/n)==h,n
        for a in r['units']:
            assert sha(a['completion'])==a['sha256'];c=read(a['completion']);validate(Path(a['completion']).parent,c['request'])
        print('VERIFIED_CACHED_ADDRESS_STAGE');return
    if args.preflight_only:units=[next(a for a in units if a.get('stratum')=='no_init_array')]
    refs=read(stage/'cpp_references.json') if (stage/'cpp_references.json').exists() else {};librefs=read(stage/'library_references.json') if (stage/'library_references.json').exists() else {};entries=[];summary=[];started=time.monotonic()
    with ThreadPoolExecutor(max_workers=1 if args.preflight_only else f['limits']['workers']) as pool:
        fs={pool.submit(one,a,stage,f,refs.get(a['source_sha256']),librefs.get(a['source_sha256'])):a for a in units}
        for future in as_completed(fs):
            a=fs[future];folder,c=future.result();r=c['result'];entries.append({'unit_id':a['unit_id'],'completion':str(folder/'completion.json'),'sha256':sha(folder/'completion.json')});summary.append({'unit_id':a['unit_id'],'stratum':a.get('stratum'),'potential_verdict':r['potential_verdict'],'unknown_reasons':';'.join(r.get('unknown_reasons',[])),'library_binding_status':r.get('library_binding_status'),'library_binding_failure':r.get('library_binding_failure'),'CPP_created':r.get('CPP_created',False),'CPP_reference_reused':bool(c['request'].get('cpp_reference')),'library_certificate_reused':bool(c['request'].get('library_binding_reference')),'failure':c['failure'],'exit_code':c['exit_code'],'elapsed_seconds':c['elapsed_seconds']})
            if args.preflight_only or len(entries)%16==0:print(json.dumps({'done':len(entries),'total':len(units),'failure':c['failure'],'potential':r['potential_verdict']}),flush=True)
    if args.preflight_only:
        atomic_json(stage/'preflight_receipt.json',{'status':'COMPLETED_INTERFACE_CHECK','unit':entries[0],'summary':summary[0],'outputs_reused_by_full_stage':True,'new_Eva_neural_training':0},immutable=True);return
    entries.sort(key=lambda a:a['unit_id']);summary.sort(key=lambda a:a['unit_id']);atomic_json(stage/'unit_index.json',entries,immutable=True);write_csv(stage/'summary.csv',summary)
    atomic_json(receipt,{'status':'COMPLETE_STATIC_ADDRESS_STAGE','freeze_sha256':sha(stage/'freeze.json'),'units':entries,'count':len(entries),'wall_seconds':time.monotonic()-started,'potential_count':sum(a['potential_verdict']==1 for a in summary),'failures':sum(bool(a['failure']) for a in summary),'CPP_created':sum(a['CPP_created'] for a in summary),'new_Eva_neural_training':0,'outputs':{n:sha(stage/n) for n in ['unit_index.json','summary.csv']}},immutable=True);print(json.dumps({k:read(receipt)[k] for k in ['count','potential_count','failures','CPP_created','wall_seconds']}))
if __name__=='__main__':main()

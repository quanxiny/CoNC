#!/usr/bin/env python3
"""Resumable native Joern+official features; fixed matched slices, no labels in parser.

Each worker owns a private JVM workspace. Completed cases are atomic gzip files.
Raw JSON is compressed after extraction; CPG binaries are deleted only after their
hashes have been recorded. Inputs themselves are never modified or executed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip,hashlib,json,os,signal,subprocess,time,traceback,resource
from pathlib import Path
from prepare_pilot import remove_comments

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=ROOT/'external_eval/paper_baselines/deepdfa'
DATA=ROOT/'external_eval/paper_baselines/deepwukong/data'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,separators=(',',':'))+'\n');t.replace(p)
def write_gz(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
    with gzip.open(t,'wt',compresslevel=3) as f:json.dump(x,f,separators=(',',':'))
    t.replace(p)
def read_gz(p):
    with gzip.open(p,'rt') as f:return json.load(f)

def prepare(dataset,limit=0):
    path=DATA/(dataset+'_text.jsonl.gz'); groups={}
    with gzip.open(path,'rt') as stream:
        for line in stream:
            r=json.loads(line)
            if limit and (r.get('split')!='train' or len(groups)>=limit and r['source_path'] not in groups):continue
            key=r['source_path']
            if key not in groups:
                ident=hashlib.sha256((dataset+'\0'+key+'\0'+r['sanitized_source_sha256']).encode()).hexdigest()[:24]
                groups[key]={'key':ident,'dataset':dataset,'source_case':r.get('source_case',ident),'source_original':key,
                             'source_original_sha256':r['raw_source_sha256'],'expected_clean_sha256':r['sanitized_source_sha256'],'slices':[]}
            groups[key]['slices'].append({k:v for k,v in r.items() if k not in {'slice_text','native_pdg_path','source_path'}})
    return list(groups.values()),sha(path)

def extract_one(record,source,manifest_hash):
    import native_features
    slices=[dict(r,source_line_numbers=r['source_lines']) for r in record['slices']]
    projected={**record,'source_clean':str(source),'source_clean_sha256':sha(source),'slices':slices}
    result=native_features.extract({'records':[projected],'source_manifest_sha256':manifest_hash})
    if len(result['cases'])!=1 or result['cases'][0]['status']!='native_extracted':
        raise ValueError(result.get('cases'))
    for graph,original in zip(result['graphs'],record['slices']):
        graph.update({k:v for k,v in original.items() if k not in {'source_lines','sanitized_source_sha256','raw_source_sha256'}})
    return result

def kill(process):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()

def worker(task):
    records,out_dir,manifest_hash,timeout,run_id,retain,contract=task
    for filename,digest in contract.items():
        if sha(filename)!=digest:raise ValueError('Frontend contract changed')
    out=Path(out_dir);worker_root=out/'workers'/f'{run_id}_{os.getpid()}'
    worker_root.mkdir(parents=True,exist_ok=True)
    sources=[]; by_source={}; results=[]
    for r in records:
        target=out/'cases'/r['key'][:2]/(r['key']+'.json.gz')
        if target.exists():
            prior=read_gz(target)
            if prior['source_manifest_sha256']!=manifest_hash or prior['cache_version']!=2:raise ValueError('Stale cache')
            if prior['status']=='complete' or not prior.get('retryable',False):
                results.append({'key':r['key'],'status':prior['status'],'cached':True});continue
        raw=Path(r['source_original']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=r['source_original_sha256']:
            # External source_path points to an already sanitized source by contract.
            if hashlib.sha256(raw).hexdigest()!=r['expected_clean_sha256']:raise ValueError('Original input changed')
        clean=remove_comments(raw.decode())
        if hashlib.sha256(clean.encode()).hexdigest()!=r['expected_clean_sha256']:raise ValueError('Sanitizer differs from shared frozen manifest')
        src=worker_root/(r['key']+'.c');src.write_text(clean)
        sources.append(str(src));by_source[str(src)]=(r,target)
    if not sources:return results
    request=worker_root/'request.json';write_json(request,{'files':sources})
    logfile=worker_root/'joern.log'
    java_home=BASE/'native_pilot/java_user'
    cmd=[str(HERE/'joern-v1.1.1072/joern-cli/joern'),'-J-Xmx5g','-J-XX:ActiveProcessorCount=4',
         '-J-Duser.home='+str(java_home),'--script',str(HERE/'batch_frontend.sc'),'--params','manifest='+str(request)]
    env=os.environ.copy();env.update(JAVA_HOME='/usr/lib/jvm/java-11-openjdk-amd64',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    done={};active=None;started=time.monotonic();last=started;offset=0;timed_out=None
    with logfile.open('w') as stream:
        process=subprocess.Popen(cmd,cwd=worker_root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        while process.poll() is None:
            time.sleep(.25)
            with logfile.open() as reader:
                reader.seek(offset)
                for line in reader:
                    if 'DEEPDFA_BEGIN ' in line:
                        try:active=json.loads(line.split('DEEPDFA_BEGIN ',1)[1])['file'];last=time.monotonic()
                        except Exception:pass
                    if 'DEEPDFA_DONE ' in line:
                        try:
                            x=json.loads(line.split('DEEPDFA_DONE ',1)[1]);done[x['file']]=x;active=None;last=time.monotonic()
                        except Exception:pass
                offset=reader.tell()
            if time.monotonic()-last>(timeout if active else 180):
                timed_out=active or 'JVM_STARTUP';kill(process);break
        # Capture final buffered completion lines.
        for line in logfile.read_text(errors='replace').splitlines():
            if 'DEEPDFA_DONE ' in line:
                try:x=json.loads(line.split('DEEPDFA_DONE ',1)[1]);done[x['file']]=x
                except Exception:pass
    for source in sources:
        r,target=by_source[source];begin=time.monotonic()
        receipt={'cache_version':2,'source_manifest_sha256':manifest_hash,'record':r,'status':'failed',
                 'source_clean_sha256':sha(source),'parse':done.get(source),'run_id':run_id,'retryable':False}
        try:
            if source not in done or done[source]['status']!='ok':raise RuntimeError('parse incomplete/failed; timeout='+str(timed_out))
            signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('official feature per-source timeout')))
            signal.alarm(timeout)
            try:result=extract_one(r,Path(source),manifest_hash)
            finally:signal.alarm(0)
            receipt.update(status='complete',features=result)
        except Exception:
            receipt['error']=traceback.format_exc();receipt['retryable']=source not in done and source!=timed_out
        receipt['feature_seconds']=time.monotonic()-begin
        artifacts={}
        for suffix in ['.nodes.json','.edges.json','.cpg.bin']:
            raw=Path(source+suffix)
            if raw.exists():
                info={'sha256':sha(raw),'bytes':raw.stat().st_size}
                if retain:
                    dst=out/'raw'/r['key'][:2]/(r['key']+suffix+'.gz');dst.parent.mkdir(parents=True,exist_ok=True)
                    if not dst.exists():
                        with raw.open('rb') as inf,gzip.open(dst,'wb',compresslevel=3) as outf:
                            import shutil;shutil.copyfileobj(inf,outf)
                    info.update(compressed_path=str(dst.relative_to(ROOT)),compressed_bytes=dst.stat().st_size,compressed_sha256=sha(dst))
                artifacts[suffix]=info;raw.unlink()
        receipt.update(native_artifacts=artifacts,child_maxrss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,worker_wall_seconds=time.monotonic()-started,
                       adapter_sha256={p.name:sha(p) for p in [HERE/'frontend.py',HERE/'native_features.py',HERE/'batch_frontend.sc',HERE/'get_func_graph.sc']})
        if target.exists():
            prior_target=target.with_name(target.name+'.prior_'+sha(target)[:12]);prior_target.write_bytes(target.read_bytes())
        for filename,digest in contract.items():
            if sha(filename)!=digest:raise ValueError('Frontend contract changed while running')
        write_gz(target,receipt);Path(source).unlink(missing_ok=True)
        results.append({'key':r['key'],'status':receipt['status'],'retryable':receipt['retryable'],'slices':len(r['slices']),
                        'parse_s':done.get(source,{}).get('seconds'),'feature_s':receipt['feature_seconds']})
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['source','mit','formai']);p.add_argument('--workers',type=int,default=12)
    p.add_argument('--batch-cases',type=int,default=64);p.add_argument('--limit-train-cases',type=int,default=0)
    p.add_argument('--timeout-seconds',type=int,default=90);p.add_argument('--run-id',required=True);p.add_argument('--out',default=str(BASE/'native_full'))
    args=p.parse_args()
    if not 1<=args.workers<=12:raise ValueError('CPU budget: 1..12 Joern workers, four CPU each')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    run=out/'runs'/args.run_id;run.mkdir(parents=True,exist_ok=True)
    lock=run/'RUNNING'
    if lock.exists():raise ValueError('Run ID already in use; choose a new ID to resume')
    lock.write_text(str(os.getpid()))
    records=[];hashes={}
    for dataset in args.datasets:
        rows,digest=prepare(dataset,args.limit_train_cases);records.extend(rows);hashes[dataset]=digest
    contract={str(p):sha(p) for p in [HERE/'frontend.py',HERE/'native_features.py',HERE/'batch_frontend.sc',HERE/'get_func_graph.sc',HERE/'JOERN_SOURCE_MANIFEST.json']}
    manifest_hash=hashlib.sha256(json.dumps({'data':hashes,'frontend':contract},sort_keys=True).encode()).hexdigest()
    write_json(run/'manifest.json',{'shared_manifests':hashes,'frontend_contract':contract,'manifest_sha256':manifest_hash,'cases':len(records),'slices':sum(len(r['slices']) for r in records),
                                  'workers':args.workers,'timeout_seconds':args.timeout_seconds,'batch_cases':args.batch_cases,'argv':vars(args)})
    pending=[];cached=0
    for r in records:
        path=out/'cases'/r['key'][:2]/(r['key']+'.json.gz')
        if path.exists():
            x=read_gz(path)
            if x['source_manifest_sha256']!=manifest_hash or x['cache_version']!=2:raise ValueError('Cache identity mismatch')
            if x['status']=='complete' or not x['retryable']:cached+=1;continue
        pending.append(r)
    batches=[pending[n:n+args.batch_cases] for n in range(0,len(pending),args.batch_cases)]
    start=time.monotonic();counts=Counter(cached=cached);write_json(run/'progress.json',dict(counts))
    print(json.dumps({'status':'START','cases':len(records),'cached':cached,'pending':len(pending),'batches':len(batches)}),flush=True)
    with (run/'ledger.jsonl').open('a') as ledger,ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,(batch,str(out),manifest_hash,args.timeout_seconds,args.run_id,True,contract)) for batch in batches]
        for f in as_completed(futures):
            try:results=f.result()
            except Exception:
                counts['batch_error']+=1;ledger.write(json.dumps({'batch_error':traceback.format_exc()})+'\n');ledger.flush();continue
            for r in results:counts[r['status']]+=1;ledger.write(json.dumps(r)+'\n')
            ledger.flush();progress={'counts':dict(counts),'wall_s':time.monotonic()-start,'manifest_sha256':manifest_hash}
            write_json(run/'progress.json',progress);print(json.dumps(progress),flush=True)
    write_json(run/'receipt.json',{'status':'FRONTEND_COMPLETE_WITH_EXPLICIT_COVERAGE','counts':dict(counts),'wall_s':time.monotonic()-start,'manifest_sha256':manifest_hash,
                                'records':len(records),'raw_json_retained_compressed':True,'sources_executed':False})
    lock.rename(run/'FINISHED')
if __name__=='__main__':main()

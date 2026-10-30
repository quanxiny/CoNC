#!/usr/bin/env python3
"""Native static-analyzer baselines. No model calls or dataset code execution.

One implementation, dataset configurations, immutable per-source receipts and
isolated native processes. Only saved native reports are used for statistics.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import resource
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = HERE / "static_baselines"
VENDOR = HERE / "vendor/static_tools"
CONFIG = WORK / "config.json"
TOOLS = ("cppcheck", "clang_sa", "frama_c_eva")
# Actual producer of all six completed native ledgers. The only subsequent
# runner change validates source/environment hashes before cache reuse; native
# commands, selectors, limits and analyzer execution are unchanged.
FROZEN_NATIVE_PRODUCER = "b1139abb3e8359d0191d81eb48673bc30b0457e58906d5b7134c62b2146b0bf0"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def signed(path):
    x = read(path); expected = x.pop('result_sha256')
    if digest(x) != expected: raise ValueError('receipt hash mismatch: '+str(path))
    return x


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def sign(path, value):
    value = dict(value); value.pop('result_sha256', None)
    write(path, {**value, 'result_sha256': digest(value)})


def immutable(path, value):
    if Path(path).exists():
        if signed(path) != value: raise ValueError('frozen artifact differs: '+str(path))
    else: sign(path, value)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def csv_write(path, rows):
    path = Path(path); temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['status'])
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def env_for(tool):
    env = dict(os.environ, LC_ALL='C', LANG='C', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if tool == 'clang_sa':
        env['LD_LIBRARY_PATH'] = str(VENDOR/'llvm14/usr/lib/x86_64-linux-gnu')
    return env


def binary(tool):
    import shutil
    names = {'cppcheck': ('CPPCHECK_BIN', 'cppcheck'), 'clang_sa': ('CLANG_BIN', 'clang-14'), 'frama_c_eva': ('FRAMAC_BIN', 'frama-c')}
    env_name, command = names[tool]
    if os.environ.get(env_name):
        return Path(os.environ[env_name]).resolve()
    installed = shutil.which(command)
    if installed:
        return Path(installed).resolve()
    return {'cppcheck': VENDOR/'cppcheck-2.22.0/cppcheck',
            'clang_sa': VENDOR/'llvm14/usr/bin/clang-14',
            'frama_c_eva': VENDOR/'frama33/bin/frama-c'}[tool]


def initialize():
    """Freeze only generic native settings and dataset IDs, before data scores."""
    config = {
        'schema': 'native-static-baselines-v1', 'tools': list(TOOLS),
        'machine_model': 'Linux x86_64 LP64; int32; wchar_t32; native C GNU11/C11',
        'wall_seconds': 60, 'cpu_seconds': 50, 'address_space_bytes': 3*1024**3,
        'output_file_bytes': 16*1024**2,
        'cppcheck': {'flags': ['--enable=warning,information,missingInclude', '--platform=unix64', '--std=c11',
                              '--library=posix', '--check-level=normal', '--max-configs=12', '--xml', '--xml-version=2'],
                     'target_ids': ['arrayIndexOutOfBounds', 'arrayIndexOutOfBoundsCond', 'negativeIndex',
                                    'bufferAccessOutOfBounds', 'invalidScanfFormatWidth', 'invalidscanf',
                                    'stlOutOfBounds', 'algorithmOutOfBounds', 'containerOutOfBounds',
                                    'containerOutOfBoundsIndexExpression'],
                     'scope': 'native buffer/index/library-size diagnostics; no general null/leak/arithmetic-only warnings'},
        'clang_sa': {'flags': ['--analyze', '-std=gnu11', '-Wno-everything', '-fno-color-diagnostics',
                              '-Xclang', '-analyzer-checker=core,unix.Malloc,alpha.security.ArrayBoundV2,alpha.unix.cstring.OutOfBounds',
                              '-Xclang', '-analyze-function=main', '-Xclang', '-analyzer-output=plist'],
                     'target_ids': ['alpha.security.ArrayBoundV2', 'alpha.unix.cstring.OutOfBounds'],
                     'scope': 'main entry with native inlining; pinned version has experimental bounds checkers'},
        'frama_c_eva': {'flags': ['-eva', '-machdep', 'x86_64', '-main', 'main', '-eva-slevel', '100'],
                        'target_property_kinds': ['index_bound', 'mem_access'],
                        'memory_library_functions': ['memcpy','memmove','memset','memcmp','strcpy','strncpy',
                                                     'strcat','strncat','strlen','fgets','fread','fwrite',
                                                     'wcscpy','wcsncpy','wmemcpy','wmemmove','wmemset'],
                        'scope': 'native index/memory validity plus memory-library validity preconditions; mem_access also includes null or dangling pointers'},
        'datasets': {'mit': {'manifest': 'external_eval/manifests/mit/scenes.jsonl',
                              'source_root': 'external_eval/manifests/mit', 'count':1164},
                     'formai': {'manifest':'external_eval/formai_pool/adaptation/test_inputs/prepared_scenes.jsonl',
                                'source_root':'external_eval/formai_pool/prepared', 'count':167}},
        'prediction_unit':'whole scene/program; any in-scope native target warning => positive',
        'negative':'tool completed with no selected warning; not a safety proof',
        'failure':'timeout, crash, syntax/header/serialization/analysis failure => UNKNOWN, never negative',
        'native_incomplete_paths':'native no-warning does not certify exhaustive path exploration',
        'ranking_metrics':'AP/AUROC NA for tools: no calibrated continuous decision score',
        'source_policy':'existing hash-verified sanitized whole source, original declarations/guards/helpers; no flaw lines, source execution or guessed stubs',
        'aggregation_scope':'one main-root program per dataset row; Cppcheck may also inspect same-file unused functions',
        'bootstrap':{'replicates':2000,'seed':20261018,'groups':'MIT base case / FormAI program ID; prompt families unknown'},
        'training_steps':0, 'model_or_threshold_changed':False,
    }
    immutable(CONFIG, config)
    artifacts = {}
    versions = {}
    for tool in TOOLS:
        exe = binary(tool)
        r = subprocess.run([str(exe), '-version' if tool=='frama_c_eva' else '--version'],
                           env=env_for(tool), capture_output=True, text=True, check=True)
        versions[tool] = (r.stdout+r.stderr).strip()
        artifacts[str(exe.resolve())] = sha(exe)
    for p in (VENDOR/'cppcheck-2.22.0/cfg').glob('*.cfg'): artifacts[str(p.resolve())]=sha(p)
    frama_share=Path(subprocess.check_output([str(binary('frama_c_eva')),'-print-share-path'],text=True).strip())
    for p in (frama_share/'libc').rglob('*'):
        if p.is_file(): artifacts[str(p.resolve())]=sha(p)
    for p in (VENDOR/'frama33').rglob('*.so*'):
        if p.is_file(): artifacts[str(p.resolve())]=sha(p)
    for p in (VENDOR/'llvm14/usr/lib/x86_64-linux-gnu').glob('lib*.so*'):
        if p.is_file(): artifacts[str(p.resolve())]=sha(p)
    native_environment = subprocess.check_output(['dpkg-query','-W','-f=${Package} ${Version}\n',
                                                 'libc6','libc6-dev','libstdc++6','gcc-11'],text=True)
    immutable(WORK/'tool_registry.json', {'versions':versions,'artifacts':artifacts,
              'downloads':{str(p.relative_to(ROOT)):sha(p) for p in (VENDOR/'downloads').glob('*.json')},
              'native_environment':native_environment,'system_preprocessor':{'path':'/usr/bin/gcc','sha256':sha('/usr/bin/gcc')},
              'official_install_scripts_executed':False})
    print(json.dumps({'status':'FROZEN_CONFIG_AND_TOOLS','versions':versions,'tracked_tool_files':len(artifacts)}))


def verify_tools():
    registry = signed(WORK/'tool_registry.json')
    for name, expected in registry['artifacts'].items():
        if sha(name)!=expected: raise ValueError('native tool dependency changed: '+name)
    if sha('/usr/bin/gcc')!=registry['system_preprocessor']['sha256']: raise ValueError('preprocessor changed')
    environment=subprocess.check_output(['dpkg-query','-W','-f=${Package} ${Version}\n',
                                          'libc6','libc6-dev','libstdc++6','gcc-11'],text=True)
    if environment!=registry['native_environment']: raise ValueError('native system library environment changed')
    return registry


def command(tool, source, directory, config):
    cmd = [str(binary(tool)), *config[tool]['flags']]
    if tool=='clang_sa': cmd += ['-o', str(directory/'native.plist'), str(source)]
    elif tool=='frama_c_eva': cmd += [str(source), '-then', '-report-csv', str(directory/'native.tsv')]
    else: cmd += [str(source)]
    return cmd


def parse_native(tool, directory, config):
    """Native report selectors are frozen from tool docs and independent fixtures."""
    text = (directory/'stdout.txt').read_text(errors='replace')+'\n'+(directory/'stderr.txt').read_text(errors='replace')
    selected, others = [], []
    if tool=='cppcheck':
        try: root = ET.fromstring((directory/'stderr.txt').read_text())
        except ET.ParseError as exc: return [], [], 'REPORT_PARSE_FAILURE:'+str(exc)
        for error in root.findall('.//error'):
            code=error.get('id'); locations=[dict(x.attrib) for x in error.findall('location')]
            row={**error.attrib,'locations':locations}
            (selected if code in config[tool]['target_ids'] else others).append(row)
        failed = {r['id'] for r in others} & {'syntaxError','internalError','preprocessorErrorDirective',
                    'unknownMacro','missingInclude','missingIncludeSystem','toomanyconfigs'}
        # Standard-library headers are intentionally handled by Cppcheck's
        # shipped native models; user headers are not replaced by invented stubs.
        failed.discard('missingIncludeSystem')
        if failed: return selected, others, 'NATIVE_INCOMPLETE:'+','.join(sorted(failed))
    elif tool=='clang_sa':
        path=directory/'native.plist'
        if not path.exists(): return [], [], 'MISSING_NATIVE_REPORT'
        try: report=plistlib.loads(path.read_bytes())
        except Exception as exc:return [], [], 'REPORT_PARSE_FAILURE:'+str(exc)
        for item in report.get('diagnostics',[]):
            row={k:item.get(k) for k in ('check_name','type','description','location')}
            row['native_files']=report.get('files',[])
            (selected if item.get('check_name') in config[tool]['target_ids'] else others).append(row)
        if re.search(r'(fatal error:|\berror:|reached.*max.*nodes|maximum.*node.*count)',text,re.I):
            return selected, others, 'NATIVE_PARSE_OR_BUDGET_FAILURE'
    else:
        path=directory/'native.tsv'
        if not path.exists():return [], [], 'MISSING_NATIVE_REPORT'
        with path.open(newline='') as stream:
            rows=list(csv.DictReader(stream,delimiter='\t'))
        for row in rows:
            if row.get('status') not in ('Unknown','Invalid','Invalid or unreachable'):continue
            kind=row.get('property kind','');prop=row.get('property','')
            is_target=kind in config[tool]['target_property_kinds']
            if kind.startswith('precondition of '):
                function=kind.removeprefix('precondition of ')
                is_target=function in config[tool]['memory_library_functions'] and bool(re.search(r'valid|readable|writable|buffer|size',prop,re.I))
            (selected if is_target else others).append(row)
        if not re.search(r'\[eva:summary\].*ANALYSIS SUMMARY',text):return selected,others,'EVA_ANALYSIS_NOT_COMPLETED'
        if re.search(r'\[(kernel|eva)(?::[^\]]+)?\].*(?:User Error|Failure|Abort|not supported|no body|Neither code nor specification)',text,re.I):
            return selected,others,'EVA_CONTEXT_OR_ANALYSIS_FAILURE'
    return selected, others, None


def run_unit(tool, item, directory, config, identity):
    source=Path(item['source_path']); directory.mkdir(parents=True,exist_ok=True)
    req={'tool':tool,'scene_id':item['scene_id'],'source_sha256':item['source_sha256'],**identity}
    request_path=directory/'request.json'; receipt_path=directory/'receipt.json'
    immutable(request_path,req)
    if sha(source)!=item['source_sha256']:raise ValueError('source changed')
    if receipt_path.exists():
        receipt=signed(receipt_path)
        if receipt['request']!=req:raise ValueError('cached request differs')
        for name,h in receipt['artifacts'].items():
            if sha(directory/name)!=h:raise ValueError('cached native output changed')
        return receipt,True
    cmd=command(tool,source,directory,config)
    started=time.monotonic(); failure=None
    with (directory/'stdout.txt').open('wb') as out,(directory/'stderr.txt').open('wb') as err:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'limited-exec',json.dumps(config),json.dumps(cmd)],
                               stdout=out,stderr=err,env=env_for(tool),start_new_session=True,cwd=ROOT)
        try:code=child.wait(timeout=config['wall_seconds'])
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait();code=-signal.SIGKILL;failure='WALL_TIMEOUT'
    elapsed=time.monotonic()-started
    selected,others,parse_failure=parse_native(tool,directory,config)
    if failure is None and code!=0:
        failure='CPU_RESOURCE_LIMIT' if code==-signal.SIGXCPU else 'NATIVE_CRASH_OR_RESOURCE_LIMIT' if code<0 else 'NATIVE_NONZERO_EXIT'
    failure=failure or parse_failure
    row={'status':'UNKNOWN' if failure else 'PASS','prediction':None if failure else int(bool(selected)),
         'failure_reason':failure,'target_alert_count':len(selected),'other_alert_count':len(others),
         'selected_alerts':selected,'other_alerts':others,'exit_code':code,'elapsed_seconds':elapsed,
         'command':cmd,'request':req,'no_dataset_program_execution':True,'training_steps':0}
    row['artifacts']={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name not in ('request.json','receipt.json')}
    sign(receipt_path,row)
    return row,False


def manifest():
    config=signed(CONFIG); rows=[]
    for dataset,spec in config['datasets'].items():
        records=jsonl(ROOT/spec['manifest'])
        assert len(records)==spec['count']
        for row in records:
            source=ROOT/spec['source_root']/row['sanitized_path']
            assert source.is_file() and sha(source)==row['sanitized_sha256']
            assert row['owned_entry']=='main' and row['label'] in (0,1)
            rows.append({'dataset':dataset,'scene_id':row['scene_id'],'label':row['label'],
                         'family_id':row['family_id'],'variant':row.get('variant'),
                         'source_path':str(source.resolve()),'source_sha256':row['sanitized_sha256'],
                         'development_diagnostic':bool(row.get('development_diagnostic') or row.get('adapter_development_diagnostic'))})
    path=WORK/'manifest.jsonl'
    data=''.join(json.dumps(r,sort_keys=True)+'\n' for r in sorted(rows,key=lambda r:(r['dataset'],r['scene_id'])))
    if path.exists():
        if path.read_text()!=data:raise ValueError('static baseline manifest changed')
    else:path.write_text(data)
    immutable(WORK/'manifest_receipt.json',{'config_sha256':sha(CONFIG),'manifest_sha256':sha(path),
              'source_manifests':{spec['manifest']:sha(ROOT/spec['manifest']) for spec in config['datasets'].values()},
              'counts':dict(Counter(r['dataset'] for r in rows)),'source_rewrites':0,'label_or_flaw_line_selection':False})
    print(json.dumps({'status':'FROZEN_SOURCE_MANIFEST','programs':len(rows)}))


def acceptance():
    config=signed(CONFIG);verify_tools()
    fixtures=sorted((WORK/'fixtures').glob('*.c'));results=[]
    identity={'config_sha256':sha(CONFIG),'tool_registry_sha256':sha(WORK/'tool_registry.json'),
              'execution_script_sha256':FROZEN_NATIVE_PRODUCER}
    for tool in TOOLS:
        for source in fixtures:
            item={'scene_id':'hand:'+source.stem,'source_path':str(source.resolve()),'source_sha256':sha(source)}
            directory=WORK/'acceptance'/digest(identity)[:16]/tool/source.stem
            receipt,cached=run_unit(tool,item,directory,config,identity)
            results.append({'tool':tool,'fixture':source.stem,'status':receipt['status'],'prediction':receipt['prediction'],
                            'target_alerts':receipt['target_alert_count'],'failure_reason':receipt['failure_reason']})
    for tool in TOOLS:
        def find(name):return next(r for r in results if r['tool']==tool and r['fixture']==name)
        assert find('array_upper')['prediction']==1 and find('array_lower')['prediction']==1
        assert find('array_safe')['prediction']==0 and find('guard_safe')['prediction']==0
        assert find('parse_failure')['status']=='UNKNOWN' and find('missing_header')['status']=='UNKNOWN'
    # A no-warning report is a detector negative, not a proof; other fixtures
    # document native capability rather than forcing every analyzer to agree.
    fake=[sys.executable,'-c','import time; time.sleep(5)']
    start=time.monotonic();proc=subprocess.Popen(fake,start_new_session=True)
    try:proc.wait(timeout=.05);raise AssertionError('timeout fixture did not time out')
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    csv_write(WORK/'acceptance.csv',results)
    immutable(WORK/'acceptance_receipt.json',{'status':'PASS','identity':identity,
              'fixtures':{str(p):sha(p) for p in fixtures},'results_sha256':sha(WORK/'acceptance.csv'),
              'checks':['array upper/lower','safe/guard','parse/header UNKNOWN','isolated timeout','native capability differences retained'],
              'dataset_sources_read':0,'dataset_code_executed':0,'native_fixture_runs':len(results)})
    print(json.dumps({'status':'PASS_NATIVE_ACCEPTANCE','runs':len(results),'timeout_test_seconds':time.monotonic()-start}))


def batch(workers):
    config=signed(CONFIG);verify_tools();accept=signed(WORK/'acceptance_receipt.json')
    if accept['identity']['execution_script_sha256']!=FROZEN_NATIVE_PRODUCER:raise ValueError('acceptance uses different runner')
    manifest_receipt=signed(WORK/'manifest_receipt.json')
    if sha(WORK/'manifest.jsonl')!=manifest_receipt['manifest_sha256']:raise ValueError('manifest changed')
    records=jsonl(WORK/'manifest.jsonl'); index=[]
    WORK.mkdir(exist_ok=True)
    with (WORK/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for dataset in config['datasets']:
            selected=[r for r in records if r['dataset']==dataset]
            for tool in TOOLS:
                identity={'config_sha256':sha(CONFIG),'tool_registry_sha256':sha(WORK/'tool_registry.json'),
                          'execution_script_sha256':FROZEN_NATIVE_PRODUCER,'manifest_receipt_sha256':sha(WORK/'manifest_receipt.json'),
                          'acceptance_receipt_sha256':sha(WORK/'acceptance_receipt.json'),'dataset':dataset}
                run_id=digest({'tool':tool,**identity});directory=WORK/'runs'/run_id
                directory.mkdir(parents=True,exist_ok=True)
                immutable(directory/'request.json',{'tool':tool,**identity})
                started=time.monotonic();completed=[];hits=0
                def execute(item):
                    unit_dir=directory/digest(item['scene_id'])[:24]
                    receipt,cached=run_unit(tool,item,unit_dir,config,identity)
                    return item,receipt,cached
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    pending=[pool.submit(execute,item) for item in selected]
                    for future in as_completed(pending):
                        item,receipt,cached=future.result();hits+=int(cached)
                        completed.append({**{k:item[k] for k in ('dataset','scene_id','label','family_id','variant','development_diagnostic')},
                            'tool':tool,**{k:receipt[k] for k in ('status','prediction','failure_reason','target_alert_count','other_alert_count','elapsed_seconds','exit_code')},
                            'unit_receipt':str((directory/digest(item['scene_id'])[:24]/'receipt.json').relative_to(ROOT))})
                        if len(completed)%100==0:
                            print(json.dumps({'dataset':dataset,'tool':tool,'completed':len(completed),'total':len(selected),'cached':hits}),flush=True)
                completed.sort(key=lambda r:r['scene_id']);csv_write(directory/'predictions.csv',completed)
                receipt={'status':'COMPLETE_LEDGER','request':{'tool':tool,**identity},'run_id':run_id,
                         'registered':len(selected),'counts':dict(Counter(r['status'] for r in completed)),
                         'failure_reasons':dict(Counter(r['failure_reason'] for r in completed if r['failure_reason'])),
                         'predictions_sha256':sha(directory/'predictions.csv')}
                immutable(directory/'run_receipt.json',receipt)
                index.append({'dataset':dataset,'tool':tool,'run_id':run_id,'directory':str(directory.relative_to(ROOT)),
                              'receipt_sha256':sha(directory/'run_receipt.json')})
                print(json.dumps({'dataset':dataset,'tool':tool,'counts':receipt['counts'],'cached':hits,
                                  'elapsed_seconds_this_invocation':time.monotonic()-started}),flush=True)
        immutable(WORK/'index.json',{'status':'COMPLETE','runs':index,'config_sha256':sha(CONFIG)})


def verify():
    verify_tools();index=signed(WORK/'index.json');count=0
    for item in jsonl(WORK/'manifest.jsonl'):
        assert sha(item['source_path'])==item['source_sha256']
    for run in index['runs']:
        directory=ROOT/run['directory'];receipt=signed(directory/'run_receipt.json')
        assert sha(directory/'run_receipt.json')==run['receipt_sha256']
        assert sha(directory/'predictions.csv')==receipt['predictions_sha256']
        with (directory/'predictions.csv').open() as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==receipt['registered']
        for row in rows:
            unit=ROOT/row['unit_receipt'];r=signed(unit)
            assert r['request']['execution_script_sha256']==FROZEN_NATIVE_PRODUCER
            for name,h in r['artifacts'].items():assert sha(unit.parent/name)==h
            count+=1
    print(json.dumps({'status':'PASS_STATIC_BASELINE_LEDGERS','runs':len(index['runs']),'unit_receipts':count,'model_calls':0}))


def main():
    if len(sys.argv)>1 and sys.argv[1]=='limited-exec':
        config,cmd=json.loads(sys.argv[2]),json.loads(sys.argv[3])
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        resource.setrlimit(resource.RLIMIT_CPU,(config['cpu_seconds'],config['cpu_seconds']+1))
        resource.setrlimit(resource.RLIMIT_AS,(config['address_space_bytes'],)*2)
        resource.setrlimit(resource.RLIMIT_FSIZE,(config['output_file_bytes'],)*2)
        os.execv(cmd[0],cmd)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('initialize','manifest','acceptance','run','verify'))
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.action=='initialize':initialize()
    elif args.action=='manifest':manifest()
    elif args.action=='acceptance':acceptance()
    elif args.action=='run':batch(args.workers)
    else:verify()


if __name__=='__main__':main()

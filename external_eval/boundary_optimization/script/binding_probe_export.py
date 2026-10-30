"""Bounded Frama-C 33 real-server access/obligation probe, not a classifier.

Uses the installed server's documented API and upstream socket framing. Never
parses prettyprinted values into integers, emulates Eva, or changes old adapters.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import time
import uuid
from evidence_char_extension import CharacterAnalyzer
from evidence_access import walk

ROOT=Path(__file__).resolve().parents[3]
TOOL=ROOT/'external_eval/vendor/static_tools/frama33/bin/frama-c'
SHARE=TOOL.parents[1]/'share/frama-c/share'
OLD=ROOT/'external_eval/tifs_runs/boundary_evidence_20261021'
CHAR=ROOT/'external_eval/tifs_runs/boundary_refinement_20261021/adapter_acceptance'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
def child_limits():
    resource.setrlimit(resource.RLIMIT_CPU,(50,50))
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))


class Client:
    """AF_UNIX framing implemented from actual Frama-C 33 server_socket.ml."""
    def __init__(self,sock,deadline):
        self.sock=sock;self.deadline=deadline;self.seq=0;self.transcript=[]

    def send(self,data):
        raw=json.dumps(data,separators=(',',':')).encode()
        header=('S%03x'%len(raw)) if len(raw)<0xfff else ('L%07x'%len(raw))
        self.sock.sendall(header.encode()+raw)

    def exact(self,n):
        b=b''
        while len(b)<n:
            self.sock.settimeout(max(0.01,self.deadline-time.monotonic()))
            x=self.sock.recv(n-len(b))
            if not x:raise EOFError('server socket closed')
            b+=x
        return b

    def receive(self):
        h=self.exact(1).decode();lengths={'S':3,'L':7,'W':15}
        if h not in lengths:raise ValueError('invalid server frame')
        n=int(self.exact(lengths[h]),16)
        if n>32*1024**2:raise ValueError('server message exceeds finite probe cap')
        return json.loads(self.exact(n))

    def rpc(self,request,data=None,cmd='GET'):
        self.seq+=1;uid=str(self.seq)
        q={'cmd':cmd,'id':uid,'request':request,'data':data};self.send(q)
        while True:
            r=self.receive()
            if isinstance(r,dict) and r.get('id')==uid:
                self.transcript.append({'query':q,'response':r})
                return {'data':r.get('data'),'error':r.get('msg'),'res':r.get('res')}


def source_inventory(source,unit_id,entry):
    abi=json.loads((OLD/'numeric_acceptance/abi.json').read_text())
    config={'character_contract':json.loads((CHAR/'character_contract.json').read_text())}
    a=CharacterAnalyzer(source,unit_id,'',abi,entry,None,config)
    sidecar=a.run();targets=[];entry_found=False
    for f in a.tree.root_node.named_children:
        if f.type!='function_definition':continue
        decl=a.field(f,'declarator');name=a.field(decl,'declarator')
        if not name or a.text(name)!=entry:continue
        entry_found=True
        for n in walk(a.field(f,'body')):
            if n.type=='subscript_expression':
                idx=a.field(n,'index');arg=a.field(n,'argument')
                targets.append({'source_access_id':unit_id+':'+str(n.start_byte)+':'+str(n.end_byte),
                    'source_span':a.span(n),'source_text':a.text(n),'kind':'array_index',
                    'role':a.access_mode(n,a.formation(n)),'array_expression':a.text(arg),'index_expression':a.text(idx),
                    'query_position':{'line':n.start_point.row+1,'column':n.start_point.column},
                    'index_query_position':{'line':idx.start_point.row+1,'column':idx.start_point.column},
                    'unevaluated':a.unevaluated(n),'formation':a.formation(n)})
            elif n.type=='call_expression':
                fn=a.text(a.field(n,'function'));args=a.field(n,'arguments').named_children
                if fn not in ('memset','memcpy'):continue
                roles=[(0,'copy_destination')]+([(1,'copy_source')] if fn=='memcpy' else [])
                for pos,role in roles:
                    if len(args)!=3:continue
                    arg=args[pos]
                    targets.append({'source_access_id':unit_id+':'+str(n.start_byte)+':'+str(n.end_byte)+':'+role,
                        'source_span':a.span(n),'source_text':a.text(n),'kind':'memory_call','role':role,
                        'callee':fn,'pointer_expression':a.text(arg),'count_expression':a.text(args[2]),
                        'query_position':{'line':n.start_point.row+1,'column':n.start_point.column},
                        'pointer_query_position':{'line':arg.start_point.row+1,'column':arg.start_point.column},
                        'index_query_position':{'line':args[2].start_point.row+1,'column':args[2].start_point.column},
                        'unevaluated':a.unevaluated(n),'formation':False})
    return targets,{'provenance':'frozen finite source AST+GCC ABI sidecar; not native-Eva type/interval export',
                    'ABI_sha256':abi['sha256'],'objects':sidecar['objects'],'source_access_facts':sidecar['source_accesses'],
                    'entry_found':entry_found,'scope':'raw AST named entry body only; not a transitive callgraph inventory',
                    'raw_AST_has_error':a.tree.root_node.has_error,
                    'preprocessor_nodes_present':any(n.type.startswith('preproc_') for n in walk(a.tree.root_node)),
                    'inventory_completeness':'UNKNOWN_PREPROCESSING_AND_TRANSITIVE_TARGETS_NOT_CERTIFIED',
                    'full_C_program_safety':'UNKNOWN'}


def export_fixture(source,unit_id,folder,entry='main'):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    identity={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'unit_id':unit_id,'entry':entry,
        'exporter_sha256':sha(__file__),'parent_access_sha256':sha(Path(__file__).parent/'evidence_access.py'),
        'parent_character_sha256':sha(Path(__file__).parent/'evidence_char_extension.py'),
        'parent_numeric_sha256':sha(Path(__file__).parent/'evidence_numeric.py'),
        'ABI_file_sha256':sha(OLD/'numeric_acceptance/abi.json'),'character_contract_file_sha256':sha(CHAR/'character_contract.json'),
        'tool_sha256':sha(TOOL),'machdep_sha256':sha(SHARE/'machdeps/machdep_x86_64.yaml'),
        'analysis_options':['x86_64','entry-before-RTE','lib-entry-if-not-main','cpp-H-before-RTE','unsafe-arrays',
                            'no-auto-builtins','RTE-mem-trivial-no-use-Eva','Eva-slevel100','AF_UNIX-same-session']}
    completion=folder/'completion.json'
    if completion.exists():
        cache=json.loads(completion.read_text())
        if cache['identity']!=identity:raise RuntimeError('cache identity mismatch; use a new controlled attempt directory')
        for name,h in cache['artifacts'].items():
            if sha(folder/name)!=h:raise RuntimeError('cached artifact mismatch:'+name)
        return json.loads((folder/'export.json').read_text())
    if (folder/'request.json').exists() or (folder/'export.json').exists():
        raise RuntimeError('uncommitted/exploratory attempt exists; refuse overwrite; use a new attempt directory')
    source_path=folder/'input.c'
    if source_path.exists():assert source_path.read_text()==source
    else:source_path.write_text(source)
    try:
        targets,source_facts=source_inventory(source,unit_id,entry)
    except Exception as exc:
        targets=[];source_facts={'status':'UNKNOWN','failure':type(exc).__name__+':'+str(exc)}
    session=uuid.uuid4().hex;sock_path=Path('/tmp')/('fc_bind_'+session+'.sock')
    # Every AST-affecting option precedes RTE. A later cpp/main change would
    # reparse the source and lose the first stage's generated obligations.
    command=[str(TOOL),'-machdep','x86_64','-main',entry,*(['-lib-entry'] if entry!='main' else ['-no-lib-entry']),
             '-cpp-extra-args=-H','-unsafe-arrays','-eva-no-builtins-auto',str(source_path.resolve()),
             '-rte','-rte-mem','-rte-trivial-annotations','-rte-no-use-eva-results','-then','-eva','-eva-slevel','100',
             '-server-socket-domain','unix','-server-socket',str(sock_path)]
    request={'command':command,'source_sha256':sha(source_path),'entry':entry,'session_id':session,
        'tool_sha256':sha(TOOL),'machdep_sha256':sha(SHARE/'machdeps/machdep_x86_64.yaml'),
        'limits':{'wall_seconds':60,'CPU_seconds':50,'address_space_GiB':3},'dataset_program_execution':False}
    request['identity']=identity
    save(folder/'request.json',request);start=time.monotonic();deadline=start+60
    out=(folder/'stdout.txt').open('w');err=(folder/'stderr.txt').open('w')
    process=subprocess.Popen(command,stdout=out,stderr=err,cwd=ROOT,preexec_fn=child_limits,start_new_session=True)
    conn=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);client=None;failure=None;data={}
    try:
        while not sock_path.exists():
            if process.poll() is not None:raise RuntimeError('Frama-C exited before socket was available')
            if time.monotonic()>deadline:raise TimeoutError('server startup timeout')
            time.sleep(.02)
        conn.connect(str(sock_path));client=Client(conn,deadline)
        while True:
            state=client.rpc('plugins.eva.analysis.getComputationState')
            if state['data']=='computed':break
            if state['error']:raise RuntimeError(state['error'])
            if time.monotonic()>deadline:raise TimeoutError('analysis timeout')
            time.sleep(.02)
        data['analysis_state']=state
        data['project']=client.rpc('kernel.project.getCurrent')
        data['main_function']=client.rpc('kernel.ast.getMainFunction')
        main=data['main_function']['data']
        if main:
            data['printed_declaration']=client.rpc('kernel.ast.printDeclaration',main)
            data['dead_code']=client.rpc('plugins.eva.ast.getDeadCode',main)
        data['callstack_ids']=client.rpc('plugins.eva.callstack.getAllCallstacks')
        data['callstack_info']=[client.rpc('plugins.eva.callstack.getCallstackInfo',c) for c in (data['callstack_ids']['data'] or [])]
        data['properties']=client.rpc('kernel.properties.fetchStatus',10000)
        data['Eva_properties']=client.rpc('plugins.eva.ast.fetchProperties',10000)
        data['declaration_attributes']=client.rpc('kernel.ast.fetchDeclAttributes',10000)
        for target in targets:
            for field,key in [('query_position','marker'),('index_query_position','index_marker'),('pointer_query_position','pointer_marker')]:
                if field not in target:continue
                position={'file':str(source_path.resolve()),**target[field]}
                target[key]=client.rpc('kernel.ast.getMarkerAt',position)
            marker=target['marker']['data']
            if not marker:
                target['binding_state']='UNKNOWN_NO_NATIVE_MARKER';continue
            target['probe_info']=client.rpc('plugins.eva.values.getProbeInfo',marker)
            target['marker_information']=client.rpc('kernel.ast.getInformation',marker)
            if target['kind']=='memory_call':target['native_callees']=client.rpc('plugins.eva.ast.getCallees',marker)
            target['callstack_ids']=client.rpc('plugins.eva.callstack.getMarkersCallstacks',[marker])
            context_ids=target['callstack_ids']['data'] or []
            stmt=(target['probe_info'].get('data') or {}).get('stmt')
            term=target.get('index_expression') or target.get('count_expression')
            target['parsed_operand_marker']=client.rpc('kernel.ast.parseExpr',{'stmt':stmt,'term':term}) if stmt and term else None
            probe=(target.get('parsed_operand_marker') or {}).get('data') or target.get('index_marker',{}).get('data') or marker
            target['operand_marker_information']=client.rpc('kernel.ast.getInformation',probe)
            target['access_point_values']=[{'callstack':cs,'evaluation':client.rpc('plugins.eva.values.getValues',{'target':probe,'callstack':cs})} for cs in context_ids]
            target['binding_state']='NATIVE_MARKER_OBSERVED_UNIQUE_JOIN_NOT_YET_CERTIFIED'
            target['structured_numeric_value']=None
            target['structured_numeric_status']='UNKNOWN_VALUE_API_IS_DECORATED_TEXT_NOT_PARSED'
        data['marker_attributes']=client.rpc('kernel.ast.fetchMarkerAttributes',10000)
        client.send('SHUTDOWN');process.wait(timeout=max(.1,min(5,deadline-time.monotonic())))
    except Exception as exc:
        failure=type(exc).__name__+':'+str(exc)
    finally:
        conn.close()
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
        out.close();err.close()
        if sock_path.exists():sock_path.unlink()
    if client:save(folder/'server_transcript.json',client.transcript)
    headers=[]
    for line in (folder/'stderr.txt').read_text().splitlines():
        if not line.startswith('.'):continue
        candidate=line.lstrip('.').strip()
        hp=Path(candidate)
        if hp.is_file():headers.append({'path':str(hp.resolve()),'sha256':sha(hp)})
    result={'schema':'binding_probe_v1','request':request,'source_targets':targets,'source_facts_separate':source_facts,
        'actual_preprocessor_header_dependencies':list({h['path']:h for h in headers}.values()),
        'native_exports':data,'failure':failure,'exit_code':process.returncode,'elapsed_seconds':time.monotonic()-start,
        'facts_used_for_classification':False,'full_C_program_safety':'UNKNOWN','dataset_program_execution':False,
        'header_trace':'stderr.txt from actual Frama-C preprocessor -H, not a function-name trust list'}
    save(folder/'export.json',result)
    artifacts={n:sha(folder/n) for n in ['input.c','request.json','stdout.txt','stderr.txt','server_transcript.json','export.json'] if (folder/n).exists()}
    save(completion,{'identity':identity,'artifacts':artifacts,'elapsed_seconds':result['elapsed_seconds'],
                     'failure':failure,'dataset_program_execution':False})
    return result

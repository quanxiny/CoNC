"""D_trace: immutable cached finite-query-trace readout. No C/AST execution."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
SCRIPTS=Path(__file__).resolve().parent
CORE='4a768be154f14d97452366b9e88511de51e29a0b3830fb0a1421d5d3cb8c3ce1'
KERNEL='10794813125e5ab6c86c890ab8737913bde26d6835d65190c092cc3cff6d7501'
FACTORY='49738c2fe15195d763ca42fe7d27325f7d893783288b77f860b1b5ecd12c8ca6'
LINK='05e9cbdbd6322b5277ebdfbc55d29463441f57be92c04030ddb8ce659ec66354'
ABI=ROOT/'external_eval/tifs_runs/boundary_evidence_20261021/numeric_acceptance/abi.json'
ABI_FILE='9293ed67f2af8a6a9c913c9d3eca4f917e45515feb88d84266235927515f9002'
CHAR=ROOT/'external_eval/tifs_runs/boundary_refinement_20261021/adapter_acceptance/character_contract.json'
CHAR_FILE='c664d62b71888d1bf2ee12f89d56b9cfbfbf9859e15d33ca627c4088f8f0dd36'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True).encode()).hexdigest()
def require(ok,reason):
    if not ok:raise ValueError(reason)
def span(s):return (s['start_byte'],s['end_byte'])
def integer(x):return isinstance(x,int) and not isinstance(x,bool)


class TraceReader:
    def __init__(self,stage_freeze_path):
        self.freeze_path=Path(stage_freeze_path);self.failure=None
        try:
            self.freeze=json.loads(self.freeze_path.read_text());self.freeze_sha=sha(self.freeze_path)
            require(self.freeze['core_sha256']==CORE,'STAGE_CORE_NOT_PINNED')
            require(self.freeze['factory_sha256']==FACTORY,'STAGE_FACTORY_NOT_PINNED')
            require(integer(self.freeze['expression_budget']) and 1<=self.freeze['expression_budget']<=4096,'STAGE_BUDGET_INVALID')
            for name,h in [('address_domain.py',CORE),('evidence_numeric.py',KERNEL),('address_library_binding.py',FACTORY),('query_access_linkage.py',LINK)]:
                key=str((SCRIPTS/name).relative_to(ROOT))
                require(self.freeze['files'].get(key)==h and sha(SCRIPTS/name)==h,'PINNED_DEPENDENCY_MISMATCH:'+name)
            require(sha(ABI)==ABI_FILE and sha(CHAR)==CHAR_FILE,'ABI_OR_CHARACTER_CONTRACT_MISMATCH')
            self.abi=json.loads(ABI.read_text())
            require(self.abi['sha256']==digest({k:v for k,v in self.abi.items() if k!='sha256'}),'ABI_INTERNAL_HASH_MISMATCH')
        except Exception as exc:self.failure=type(exc).__name__+':'+str(exc)

    def library(self,folder,request,result,source_sha):
        status=result.get('library_binding_status')
        if status=='NOT_REQUIRED_NO_INCLUDE_OR_LIBRARY_CALL':
            require(result.get('library_binding_sha256') is None,'UNEXPECTED_LIBRARY_HASH')
            require(request.get('library_binding_reference') is None,'UNEXPECTED_LIBRARY_REFERENCE')
            return None
        require(status in ('BOUND','SOURCE_CERTIFICATE_REUSED'),'LIBRARY_NOT_BOUND')
        ref=request.get('library_binding_reference')
        if ref:
            require(result.get('library_certificate_reference')==ref,'LIBRARY_REFERENCE_DIFFERS')
            require(ref.get('result_path') in ref['files'],'LIBRARY_RESULT_NOT_HASHED')
            for p,h in ref['files'].items():require(sha(p)==h,'LIBRARY_REFERENCE_ARTIFACT_MISMATCH')
            prepared=json.loads(Path(ref['result_path']).read_text())
        else:
            lib=folder/'library_binding';c=json.loads((lib/'completion.json').read_text())
            require(c['identity']['implementation_sha256']==FACTORY,'LIBRARY_FACTORY_IDENTITY_MISMATCH')
            require(c['identity']['source_sha256']==source_sha,'LIBRARY_COMPLETION_SOURCE_MISMATCH')
            for p,h in c['artifacts'].items():
                path=(lib/p).resolve();require(path.is_relative_to(lib.resolve()),'LIBRARY_ARTIFACT_PATH_ESCAPE')
                require(sha(path)==h,'LIBRARY_ARTIFACT_MISMATCH')
            prepared=json.loads((lib/'result.json').read_text())
        require(prepared.get('status')=='BOUND' and not prepared.get('failure'),'LIBRARY_PREPARATION_FAILED')
        b=prepared['binding'];require(b['sha256']==digest({k:v for k,v in b.items() if k!='sha256'}),'LIBRARY_DIGEST_MISMATCH')
        require(b['source_sha256']==source_sha,'LIBRARY_SOURCE_MISMATCH')
        require(b['binding_extractor_sha256']==FACTORY,'LIBRARY_EXTRACTOR_MISMATCH')
        require(b['sha256']==result.get('library_binding_sha256'),'LIBRARY_RESULT_HASH_MISMATCH')
        require(sha(b['source_CPP_file'])==b['source_CPP_sha256'],'LIBRARY_CPP_MISMATCH')
        require(sha(b['compiler'])==b['compiler_sha256']==self.abi['compiler_sha256'],'LIBRARY_COMPILER_MISMATCH')
        for p,h in b['headers'].items():require(sha(p)==h,'LIBRARY_HEADER_MISMATCH')
        return b

    def read(self,completion_path,registration,expected_completion_sha256):
        out={'schema':'D_trace_v1','D_trace':'UNKNOWN','D_pos':'UNKNOWN',
            'scope':'registered query bounds on the completed finite source-address model trace only',
            'all_C_paths_covered':False,'full_C_program_safety':'UNKNOWN','reachability':'NOT_PROVED',
            'source_or_model_reexecution':False,'readout_sha256':sha(__file__),'reasons':[]}
        try:
            require(self.failure is None,'STAGE_IDENTITY_FAILED:'+str(self.failure))
            cp=Path(completion_path).resolve();folder=cp.parent
            require(sha(cp)==expected_completion_sha256,'COMPLETION_INDEX_HASH_MISMATCH')
            c=json.loads(cp.read_text());request=c['request'];r=c['result']
            require(c.get('exit_code')==0 and not c.get('failure'),'WORKER_NOT_SUCCESS')
            require({'request.json','result.json'}.issubset(c['artifacts']),'REQUIRED_ARTIFACT_HASH_MISSING')
            for name,h in c['artifacts'].items():
                p=(folder/name).resolve();require(p.is_relative_to(folder),'ARTIFACT_PATH_ESCAPE')
                require(sha(p)==h,'UNIT_ARTIFACT_MISMATCH:'+name)
            require(json.loads((folder/'request.json').read_text())==request,'REQUEST_EMBEDDED_COPY_DIFFERS')
            require(json.loads((folder/'result.json').read_text())==r,'RESULT_EMBEDDED_COPY_DIFFERS')
            require(request['stage_freeze_sha256']==self.freeze_sha,'STAGE_FREEZE_HASH_MISMATCH')
            for k in ('unit_id','source_sha256','entry','query_registration'):
                require(request[k]==registration[k],'REGISTERED_IDENTITY_MISMATCH:'+k)
            require(Path(request['source_path']).resolve()==Path(registration['source_path']).resolve(),'SOURCE_PATH_MISMATCH')
            raw=Path(registration['source_path']).read_bytes();source_sha=hashlib.sha256(raw).hexdigest()
            require(source_sha==registration['source_sha256']==r['source_sha256'],'SOURCE_CONTENT_HASH_MISMATCH')
            require(r['unit_id']==registration['unit_id'],'RESULT_UNIT_MISMATCH')
            require(r.get('implementation_sha256')==CORE and r.get('typed_kernel_sha256')==KERNEL,'RESULT_IMPLEMENTATION_NOT_PINNED')
            require(r.get('schema')=='source_address_domain_v1' and r.get('not_native_Eva_facts') is True and r.get('source_execution') is False,'RESULT_SCOPE_NOT_PINNED')
            require(r.get('full_C_program_safety')=='UNKNOWN' and r.get('reachability')=='NOT_PROVED','RESULT_SCOPE_PROMOTED')
            require(r.get('ABI_sha256')==self.abi['sha256'],'RESULT_ABI_MISMATCH')
            require(not r.get('failure') and not r.get('library_binding_failure'),'RESULT_FAILURE')
            link=r['query_linkage'];q=registration['query_registration'];target=registration['query_span']
            require(link.get('linkage_status')=='DIRECT_SYNTAX_UNIQUE' and link.get('entry_function_AST_valid') is True,'QUERY_NOT_DIRECT_UNIQUE')
            for k in ('source_sha256','unit_id','entry'):require(link[k]==registration[k],'LINK_IDENTITY_DIFFERS:'+k)
            for k in ('source_case','query_line','row','family_id'):require(link.get(k)==q.get(k),'LINK_REGISTRATION_DIFFERS:'+k)
            a=link['unique_access'];require(len(link['candidates'])==1 and link['candidates'][0]==a,'QUERY_CANDIDATE_SET_DIFFERS')
            require(a.get('evaluated_syntax_candidate') is True and not a.get('excluded_reasons') and not a.get('semantic_caveats'),'QUERY_NOT_EVALUATED_SUPPORTED_ACCESS')
            require(a['kind'] in ('array_index','pointer_dereference') and len(a['roles'])==1,'QUERY_ROLE_UNSUPPORTED')
            start,end=span(target);require(integer(start) and integer(end) and 0<=start<end<=len(raw),'QUERY_SPAN_INVALID')
            require(span(a['source_span'])==span(r['query_span'])==(start,end),'REGISTERED_QUERY_SPAN_DIFFERS')
            expected_id=registration['unit_id']+':'+str(start)+':'+str(end)
            require(registration['source_access_id']==link['source_access_id']==expected_id,'REGISTERED_ACCESS_ID_DIFFERS')
            require(a['source_access_ids']==[expected_id],'QUERY_SOURCE_IDS_DIFFERS')
            require(raw[start:end].decode()==a['source_text'],'QUERY_SOURCE_TEXT_MISMATCH')
            require(a['source_span']['start_line']==raw[:start].count(b'\n')+1 and a['source_span']['end_line']==raw[:end].count(b'\n')+1,'QUERY_SOURCE_LINE_METADATA_DIFFERS')
            require(a['source_span']['start_line']<=q['query_line']<=a['source_span']['end_line'],'QUERY_LINE_NOT_IN_REGISTERED_SPAN')
            self.library(folder,request,r,source_sha)
            trace=r.get('trace');steps=r.get('steps');budget=r.get('budget')
            require(integer(steps) and integer(budget) and budget==request['expression_budget']==self.freeze['expression_budget'],'BUDGET_IDENTITY_MISMATCH')
            require(isinstance(trace,list) and 0<steps==len(trace)<=budget<=4096,'TRACE_LENGTH_OR_BUDGET_INVALID')
            require(all(integer(t.get('step')) and t['step']==i for i,t in enumerate(trace,1)),'TRACE_STEPS_NOT_SEQUENTIAL')
            queries=[t for t in trace if t.get('kind')=='query_address_evaluation']
            require(queries and all(span(t['span'])==(start,end) for t in queries),'QUERY_NOT_VISITED_OR_TICK_SPAN_DIFFERS')
            out.update({'unit_id':registration['unit_id'],'source_sha256':source_sha,'entry':registration['entry'],
                'query_span':target,'source_access_id':expected_id,'query_visit_count':len(queries),
                'completion_path':str(cp),'completion_sha256':expected_completion_sha256,'stage_freeze_sha256':self.freeze_sha,
                'core_sha256':CORE,'assumptions':r.get('assumptions',[]),'unknown_content_read_count':len(r.get('memory_content_reads',[])),
                'other_C_UB_preconditions':r.get('other_C_UB_preconditions','NOT_DISCHARGED')})
            if integer(r.get('potential_verdict')) and r['potential_verdict']==1:
                w=r.get('observed_address_witness');require(isinstance(w,dict),'POSITIVE_WITNESS_MISSING')
                require(span(w['source_span'])==(start,end) and w['source_text']==a['source_text'],'POSITIVE_WITNESS_TARGET_DIFFERS')
                require(w['unit']=='bytes' and w['relation']=='BOUND_VIOLATED','POSITIVE_WITNESS_RELATION_INVALID')
                off,width,cap=w['offset_bytes'],w['width_bytes'],w['capacity_bytes']
                require(all(integer(x) for x in (off,width,cap)) and width>0 and cap>0,'POSITIVE_WITNESS_NUMERIC_INVALID')
                require(not(0<=off<=cap and off+width<=cap),'POSITIVE_WITNESS_NOT_OUT_OF_BOUNDS')
                out.update({'D_trace':1,'D_pos':1,'reasons':['PRESERVED_PINNED_CORE_POTENTIAL']});return out
            require(r.get('potential_verdict')=='UNKNOWN','ORIGINAL_VERDICT_NOT_U')
            require(r.get('observed_address_witness') is None,'UNKNOWN_WITH_NONEMPTY_WITNESS')
            require(r.get('unknown_reasons')==['ENTRY_RETURNED'],'TRACE_NOT_EXACT_COMPLETED_RETURN')
            out.update({'D_trace':0,'reasons':['COMPLETED_PINNED_MODEL_TRACE_QUERY_BOUNDS_SATISFIED'],
                'conditionality':'malloc success assumptions/unknown contents/other UB retained; no claim for unvisited paths or all C executions'})
        except Exception as exc:out['reasons']=[type(exc).__name__+':'+str(exc)]
        return out

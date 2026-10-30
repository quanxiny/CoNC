"""Isolated source address-domain unit; never imports a parser in the controller."""
from pathlib import Path
import sys,json,resource,hashlib

def main():
    q=json.loads(Path(sys.argv[1]).read_text());folder=Path(sys.argv[2]);limit=q['limits']
    resource.setrlimit(resource.RLIMIT_CPU,(limit['cpu_seconds'],limit['cpu_seconds']+1));resource.setrlimit(resource.RLIMIT_AS,(limit['address_space_bytes'],limit['address_space_bytes']));resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,64*1024**2));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    raw=Path(q['source_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==q['source_sha256'];source=raw.decode('utf-8')
    from query_access_linkage import link_query
    link=link_query(source,q['query_registration'],{'source_sha256':q['source_sha256'],'entry':q['entry']})
    prefix={'unit_id':q['unit_id'],'source_sha256':q['source_sha256'],'query_linkage':link,'new_Eva_runs':0,'new_neural_training':0,'reachability':'NOT_PROVED','full_C_program_safety':'UNKNOWN','classification_scope':'conditional finite source-address potential only; no proven path or complete C proof'}
    if link['linkage_status']!='DIRECT_SYNTAX_UNIQUE':
        r={**prefix,'potential_verdict':'UNKNOWN','unknown_reasons':['QUERY_NOT_DIRECT_UNIQUE'],'library_binding_status':'NOT_ATTEMPTED','CPP_created':False}
    else:
        from address_domain import analyze_address,parse
        # Includes require a source-specific pinned header certificate too.
        tree=parse(raw);todo=[tree.root_node];need=False
        while todo:
            n=todo.pop()
            if n.type=='preproc_include':need=True
            if n.type=='call_expression':
                fn=n.child_by_field_name('function')
                if fn and raw[fn.start_byte:fn.end_byte] in [b'malloc',b'free',b'assert']:need=True
            todo.extend(n.named_children)
        binding=None;binding_status='NOT_REQUIRED_NO_INCLUDE_OR_LIBRARY_CALL';failure=None;created=False
        if need:
            from address_library_binding import prepare_library_binding
            certificate=q.get('library_binding_reference')
            if certificate:
                from address_domain import digest
                for n,h in certificate['files'].items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h,n
                b=json.loads(Path(certificate['result_path']).read_text());binding=b['binding'];assert b['status']=='BOUND'
                assert binding['source_sha256']==q['source_sha256']
                assert binding['sha256']==digest({k:v for k,v in binding.items() if k!='sha256'})
                assert binding['binding_extractor_sha256']==hashlib.sha256(Path(__file__).with_name('address_library_binding.py').read_bytes()).hexdigest()
                assert hashlib.sha256(Path(binding['source_CPP_file']).read_bytes()).hexdigest()==binding['source_CPP_sha256']
                assert hashlib.sha256(Path(binding['compiler']).read_bytes()).hexdigest()==binding['compiler_sha256']
                for n,h in binding['headers'].items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h,n
                binding_status='SOURCE_CERTIFICATE_REUSED'
            else:
                ref=q.get('cpp_reference');cpp=ref['folder'] if ref else None
                if ref:
                    for n,h in ref['files'].items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h,n
                    assert hashlib.sha256((Path(cpp)/'original.c').read_bytes()).hexdigest()==q['source_sha256']
                b=prepare_library_binding(source,q['unit_id'],folder/'library_binding',cpp_folder=cpp);binding=b.get('binding');binding_status=b['status'];failure=b.get('failure');created=(folder/'library_binding/cpp/original.i').exists()
        s=link['unique_access']['source_span'];r=analyze_address(source,q['unit_id'],query_span={'start_byte':s['start_byte'],'end_byte':s['end_byte']},entry=q['entry'],library_binding=binding,budget=q['expression_budget'])
        assert r['potential_verdict'] in [1,'UNKNOWN']
        r={**r,**prefix,'library_binding_status':binding_status,'library_binding_failure':failure,'CPP_created':created,'CPP_reused_reference':q.get('cpp_reference'),'library_certificate_reference':q.get('library_binding_reference'),'expression_budget':q['expression_budget']}
    (folder/'result.json').write_text(json.dumps(r,sort_keys=True)+'\n')
if __name__=='__main__':main()

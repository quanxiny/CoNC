"""Bounded existing-CPP standard allocator evidence; no Eva or C execution.

Header extraction runs in an isolated worker. Only two exact real prototypes
are parsed as tiny C snippets; the large header AST is never traversed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from address_domain import digest,parse
from binding_probe_export import OLD,SHARE,TOOL,child_limits,sha,save
from binding_identity import cpp_equivalence


PROTOTYPES={'externvoid*malloc(size_tsize);':'malloc','externvoidfree(void*p);':'free'}


def extract(source,folder):
    folder=Path(folder);pp=folder/'original.i';raw=pp.read_bytes()
    receipt=json.loads((folder/'cpp_original.json').read_text())
    if receipt['returncode']!=0 or receipt.get('failure'):raise ValueError('CPP_NOT_SUCCESS')
    command=receipt['command'];inputs=[Path(a) for a in command if a.endswith('.c') and Path(a).is_file()]
    if len(inputs)!=1 or inputs[0].read_bytes()!=source.encode():raise ValueError('CPP_SOURCE_IDENTITY_MISMATCH')
    abi=json.loads((OLD/'numeric_acceptance/abi.json').read_text())
    if sha(command[0])!=abi['compiler_sha256']:raise ValueError('CPP_COMPILER_ABI_MISMATCH')
    headers={};libc=(SHARE/'libc').resolve()
    for line in (folder/'cpp_original.stderr').read_text().splitlines():
        if not line.startswith('.'):continue
        p=Path(line.lstrip('.').strip())
        if not p.is_file() or '/tmp/__fc_machdep' in str(p):continue
        if not p.resolve().is_relative_to(libc):raise ValueError('HEADER_OUTSIDE_VERIFIED_BUNDLE')
        headers[str(p.resolve())]=sha(p)
    if str(libc/'stdlib.h') not in headers:raise ValueError('ACTUAL_STDLIB_NOT_OBSERVED')
    declarations={};assert_macro=None;size_t=None;current_file=None;current_line=0
    for row,line in enumerate(raw.decode().splitlines(),1):
        parts=shlex.split(line) if line.lstrip().startswith('# ') else []
        if len(parts)>=3 and parts[0]=='#' and parts[1].isdecimal():
            current_file=parts[2];current_line=int(parts[1]);continue
        compact=''.join(line.split())
        if current_file==str(libc/'stdlib.h') and compact in PROTOTYPES:
            fn=PROTOTYPES[compact]
            if fn in declarations:raise ValueError('DUPLICATE_SUPPORTED_DECLARATION')
            header_lines=Path(current_file).read_text().splitlines()
            if not 1<=current_line<=len(header_lines) or ''.join(header_lines[current_line-1].split())!=compact:
                raise ValueError('PREPROCESSED_DECLARATION_HEADER_LINE_MISMATCH')
            # Typed syntax validation is performed only on this exact small
            # snippet. Origin/hash/actual CPP proof are independently required.
            snippet=line.encode();tree=parse(snippet)
            if tree.root_node.has_error or len(tree.root_node.named_children)!=1:raise ValueError('PROTOTYPE_PARSE_FAILED')
            n=tree.root_node.named_children[0]
            if n.type!='declaration':raise ValueError('PROTOTYPE_NOT_DECLARATION')
            f=n.child_by_field_name('declarator')
            while f and f.type=='pointer_declarator':f=f.child_by_field_name('declarator')
            if f is None or f.type!='function_declarator':raise ValueError('PROTOTYPE_NOT_FUNCTION')
            name=f.child_by_field_name('declarator')
            if name is None or snippet[name.start_byte:name.end_byte].decode()!=fn:raise ValueError('PROTOTYPE_NAME_MISMATCH')
            declarations[fn]={'declaration':line,'header':current_file,'header_line':current_line,'preprocessed_line':row}
        if compact=='typedefunsignedlongsize_t;' and current_file==str(libc/'__fc_define_size_t.h'):
            size_t={'bits':64,'signed':False,'source_header':current_file,'header_line':current_line,
                'actual_typedef':line,'ABI_sha256':abi['sha256']}
        if compact=='#defineassert(e)(__FC_assert((e)!=0,__FC_FILENAME__,__LINE__,#e))' and current_file==str(libc/'assert.h'):
            ah=Path(current_file).read_text()
            if 'requires nonnull_c: c != 0;' not in ah or 'assigns \\nothing;' not in ah:
                raise ValueError('ASSERT_EFFECT_CONTRACT_NOT_VERIFIED')
            assert_macro={'header':current_file,'definition':line,'preprocessed_line':row,
                'semantics':'evaluate predicate once; nonzero precondition; assigns nothing in bundled contract'}
        current_line+=1
    if set(declarations)!=set(('malloc','free')) or size_t is None:raise ValueError('SUPPORTED_DECLARATION_OR_SIZE_T_NOT_VERIFIED')
    body={'schema':'address_stdlib_binding_v1','source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'source_CPP_file':str(pp.resolve()),'source_CPP_sha256':sha(pp),'source_CPP_receipt_sha256':sha(folder/'cpp_original.json'),
        'compiler':command[0],'compiler_sha256':sha(command[0]),'headers':headers,'declarations':declarations,
        'size_t_type':size_t,'assert_macro':assert_macro,'binding_extractor_sha256':sha(__file__),
        'scope':'actual original-source preprocessing/header origin/prototype/ABI evidence; no Eva state inferred'}
    body['sha256']=digest(body);return body


def prepare_library_binding(source,unit_id,folder,cpp_folder=None):
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    identity={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'unit_id':unit_id,
        'implementation_sha256':sha(__file__),'address_domain_sha256':sha(Path(__file__).with_name('address_domain.py')),
        'token_helper_sha256':sha(Path(__file__).with_name('binding_identity.py')),'tool_sha256':sha(TOOL),
        'ABI_file_sha256':sha(OLD/'numeric_acceptance/abi.json'),
        'environment_sha256':digest(dict(os.environ)),'provided_CPP_folder':str(Path(cpp_folder).resolve()) if cpp_folder else None}
    completion=folder/'completion.json'
    if completion.exists():
        old=json.loads(completion.read_text())
        if old['identity']!=identity:raise RuntimeError('BINDING_CACHE_IDENTITY_MISMATCH')
        for p,h in old['artifacts'].items():
            if sha(folder/p)!=h:raise RuntimeError('BINDING_CACHE_ARTIFACT_MISMATCH')
        result=json.loads((folder/'result.json').read_text())
        b=result.get('binding')
        if b and (sha(b['source_CPP_file'])!=b['source_CPP_sha256'] or any(sha(p)!=h for p,h in b['headers'].items())):
            raise RuntimeError('BINDING_CACHE_EXTERNAL_DEPENDENCY_MISMATCH')
        return result
    if any(folder.iterdir()):raise RuntimeError('BINDING_ATTEMPT_NONEMPTY_REFUSE_OVERWRITE')
    save(folder/'request.json',identity);(folder/'source.c').write_text(source)
    result={'status':'UNKNOWN','binding':None,'failure':None,'Eva_runs':0,'source_execution':False}
    try:
        if cpp_folder is None:
            pp=folder/'cpp';pp.mkdir();original=pp/'original.c';original.write_text(source)
            # This compares a source to itself solely to reuse the actual
            # Frama-generated preprocessing recipe. It is not a view proof.
            cpp_equivalence(original,original,pp,'main');cpp_folder=pp
        command=[sys.executable,str(Path(__file__).resolve()),'--extract',str(Path(cpp_folder).resolve()),
            '--source',str(folder/'source.c'),'--output',str(folder/'binding.json')]
        r=subprocess.run(command,capture_output=True,timeout=60,preexec_fn=child_limits,start_new_session=True)
        (folder/'extract.stdout').write_bytes(r.stdout);(folder/'extract.stderr').write_bytes(r.stderr)
        save(folder/'extract_receipt.json',{'command':command,'returncode':r.returncode,'limits':{'wall_seconds':60,'CPU_seconds':50,'address_space_GiB':3},'source_execution':False})
        if r.returncode!=0:raise RuntimeError('ISOLATED_EXTRACT_FAILED:'+str(r.returncode))
        result['binding']=json.loads((folder/'binding.json').read_text());result['status']='BOUND'
    except Exception as exc:result['failure']=type(exc).__name__+':'+str(exc)
    save(folder/'result.json',result)
    save(completion,{'identity':identity,'artifacts':{str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}})
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--extract',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    save(a.output,extract(a.source.read_text(),a.extract))


if __name__=='__main__':main()

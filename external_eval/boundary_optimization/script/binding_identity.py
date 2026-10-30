"""Finite whitespace-view/native identity probe. No classification or C execution.

Reuses the frozen real-server exporter. A token comparison is necessary but is
not a whole-C equivalence proof or a complete target-obligation certificate.
"""
import bisect
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import tree_sitter
import tree_sitter_c
from binding_probe_export import ROOT, TOOL, SHARE, OLD, CHAR, child_limits, export_fixture, source_inventory, sha, save


def parse(raw):
    return tree_sitter.Parser(tree_sitter.Language(tree_sitter_c.language())).parse(raw)


def nodes(n):
    yield n
    for c in n.named_children:yield from nodes(c)


def source_tree_acceptable(tree,raw):
    """Recognize only the parser's known missing include newline-token quirk."""
    def visit(n,include=None):
        if n.type=='preproc_include':include=n
        if n.type=='ERROR':return False
        if n.is_missing:
            if n.type!='preproc_include_token2' or include is None:return False
            path=include.child_by_field_name('path')
            if path is None or path.type not in ('system_lib_string','string_literal'):return False
            prefix=raw[include.start_byte:path.start_byte]
            if b''.join(prefix.split())!=b'#include':return False
            if raw[path.end_byte:include.end_byte].strip():return False
        return all(visit(c,include) for c in n.children)
    return visit(tree.root_node)


def without_location_directives(raw):
    """Remove genuine cpp #line metadata outside comments/literals only.

    This small lexical state pass does not parse C or infer any numeric fact.
    It prevents a grammar's opaque empty #define record swallowing the next
    generated #line and its irrelevant temporary filename.
    """
    out=[];block=False
    for line in raw.splitlines(keepends=True):
        parts=line.lstrip().split()
        location=not block and ((len(parts)>=3 and parts[0]==b'#' and parts[1].isdigit() and parts[2].startswith(b'"'))
            or (len(parts)>=3 and parts[0]==b'#line' and parts[1].isdigit() and parts[2].startswith(b'"')))
        if location:out.append(b'\n');continue
        out.append(line);i=0
        while i<len(line):
            if block:
                j=line.find(b'*/',i)
                if j<0:break
                block=False;i=j+2
            elif line[i:i+2]==b'/*':block=True;i+=2
            elif line[i:i+2]==b'//':break
            elif line[i:i+1] in (b'"',b"'"):
                quote=line[i];i+=1
                while i<len(line):
                    if line[i]==92:i+=2
                    elif line[i]==quote:i+=1;break
                    else:i+=1
            else:i+=1
    return b''.join(out)


def tokens(raw):
    """Exact syntax-leaf bytes after preprocessing; skip only cpp metadata.

    Comments that can be ACSL are retained, including their exact spelling.
    Parser failure is rejected, not repaired by a fallback lexer.
    """
    raw=without_location_directives(raw);tree=parse(raw)
    out=[]
    def visit(n):
        if n.type.startswith('preproc_'):
            text=raw[n.start_byte:n.end_byte].decode()
            words=text.lstrip().removeprefix('#').lstrip().split(maxsplit=1)
            directive=words[0] if words else ''
            if directive.isdecimal() or directive=='line':return
            if directive in ('define','undef','include','pragma'):
                out.append(['cpp_'+directive,text.rstrip()]);return
            raise ValueError('PREPROCESSOR_DIRECTIVE_TOKEN_UNSUPPORTED:'+directive)
        # Macro-definition records are compared byte-for-byte above and are
        # not C syntax. GNU args... in such a record can be a parser ERROR;
        # errors in the actual expanded C token stream are still rejected.
        if n.type=='ERROR' or n.is_missing:raise ValueError('TOKEN_PARSE_ERROR')
        if n.type=='comment':
            text=raw[n.start_byte:n.end_byte]
            if text.startswith((b'/*@',b'//@')):out.append(['acsl_comment',text.decode()])
            return
        if not n.children:
            if n.is_missing:raise ValueError('MISSING_TOKEN')
            out.append([n.type,raw[n.start_byte:n.end_byte].decode()]);return
        for c in n.children:visit(c)
    visit(tree.root_node)
    return out


def make_view(source):
    raw=source.encode();tree=parse(raw)
    if not source_tree_acceptable(tree,raw):raise ValueError('SOURCE_PARSE_ERROR')
    if not raw.isascii():raise ValueError('NON_ASCII_SOURCE_ENCODING_NOT_VERIFIED')
    if b'\\\n' in raw or b'\\\r\n' in raw or b'??' in raw:raise ValueError('LINE_SPLICE_OR_TRIGRAPH_UNSUPPORTED')
    for n in nodes(tree.root_node):
        if n.type.startswith('preproc_') and n.type!='preproc_include':raise ValueError('SOURCE_DIRECTIVE_OTHER_THAN_INCLUDE')
        if n.type=='comment' and raw[n.start_byte:n.end_byte].startswith((b'/*@',b'//@')):
            raise ValueError('SOURCE_ACSL_UNSUPPORTED')
        if n.type=='identifier' and raw[n.start_byte:n.end_byte] in (b'__DATE__',b'__TIME__',b'__TIMESTAMP__'):
            raise ValueError('VOLATILE_PREDEFINED_MACRO_UNSUPPORTED')
        if n.type=='preproc_include':
            path=n.child_by_field_name('path')
            if path is None or path.type not in ('system_lib_string','string_literal'):
                raise ValueError('MACRO_INCLUDE_PATH_UNSUPPORTED')
    points=set()
    for f in tree.root_node.named_children:
        if f.type!='function_definition':continue
        body=f.child_by_field_name('body')
        for n in nodes(body):
            if n.type=='compound_statement':
                points.update((n.start_byte+1,n.end_byte-1))
            elif n.type.endswith('_statement') or n.type=='declaration':
                points.update((n.start_byte,n.end_byte))
    # Inserting only new whitespace preserves all original bytes. Exact token
    # comparison below additionally checks lexical boundaries and comments.
    chunks=[];segments=[];insertions=[];cursor=0;voff=0
    for pos in sorted(points|{len(raw)}):
        if pos>cursor:
            chunk=raw[cursor:pos];chunks.append(chunk)
            segments.append({'original_start':cursor,'original_end':pos,'view_start':voff,'view_end':voff+len(chunk)})
            voff+=len(chunk)
        if pos in points:
            chunks.append(b'\n');insertions.append({'original_boundary':pos,'view_start':voff,'view_end':voff+1});voff+=1
        cursor=pos
    view=b''.join(chunks)
    if tokens(raw)!=tokens(view):raise ValueError('RAW_SOURCE_VIEW_TOKEN_MISMATCH')
    return view.decode(),{'segments':segments,'inserted_whitespace':insertions,
        'mapping':'piecewise copied original bytes; inserted newline bytes have no original identity',
        'raw_token_sequences_equal':True}


def original_offset(mapping,offset,end=False):
    # end=True maps an exclusive copied-span endpoint without consuming a
    # newline inserted immediately after it.
    for s in mapping['segments']:
        if s['view_start']<=offset<s['view_end'] or (end and offset==s['view_end']):
            return s['original_start']+offset-s['view_start']
    return None


def bounded(command,folder,name):
    start=time.monotonic()
    try:
        r=subprocess.run(command,capture_output=True,timeout=60,cwd=ROOT,preexec_fn=child_limits)
        result={'returncode':r.returncode,'failure':None};out=r.stdout;err=r.stderr
    except subprocess.TimeoutExpired as exc:
        result={'returncode':None,'failure':'TIMEOUT'};out=exc.stdout or b'';err=exc.stderr or b''
    (folder/(name+'.stdout')).write_bytes(out);(folder/(name+'.stderr')).write_bytes(err)
    result.update({'command':command,'elapsed_seconds':time.monotonic()-start,
        'limits':{'wall_seconds':60,'CPU_seconds':50,'address_space_GiB':3},'source_execution':False})
    save(folder/(name+'.json'),result)
    if result['failure'] or result['returncode']!=0:raise RuntimeError(name+':COMMAND_FAILED')
    return out.decode()


def cpp_equivalence(original,view,folder,entry):
    options=['-machdep','x86_64','-main',entry,*(['-lib-entry'] if entry!='main' else ['-no-lib-entry']),
        '-cpp-extra-args=-H','-unsafe-arrays','-eva-no-builtins-auto']
    text=bounded([str(TOOL),*options,'-keep-temp-files','-print-cpp-commands',str(original)],folder,'cpp_recipe')
    lines=[x.strip() for x in text.splitlines() if x.startswith('  ') and ' -o ' in x]
    if len(lines)!=1:raise ValueError('CPP_RECIPE_NOT_UNIQUE')
    argv=shlex.split(lines[0]);compiler=shutil.which(argv[0])
    if compiler is None or argv.count(str(original))!=1 or argv.count('-o')!=1:
        raise ValueError('CPP_RECIPE_UNSUPPORTED')
    compiler=str(Path(compiler).resolve());argv[0]=compiler
    # One actual Frama-generated recipe, environment and generated machdep
    # macro files are shared by original/view; only input/output paths change.
    macro_files={}
    for a in argv:
        if a.startswith('-I/tmp/__fc_machdep'):
            for f in Path(a[2:]).glob('*'):
                if f.is_file():macro_files[str(f)]=sha(f)
    if not macro_files:raise ValueError('FRAMA_GENERATED_MACROS_NOT_RECORDED')
    save(folder/'generated_machdep_macros.json',macro_files)
    snapshots=folder/'generated_macro_snapshots';snapshots.mkdir()
    for p in macro_files:shutil.copyfile(p,snapshots/Path(p).name)
    pp=[]
    for name,path in [('original',original),('view',view)]:
        cmd=list(argv);cmd[cmd.index(str(original))]=str(path);cmd[cmd.index('-o')+1]=str((folder/(name+'.i')).resolve())
        bounded(cmd,folder,'cpp_'+name)
        seq=tokens((folder/(name+'.i')).read_bytes());save(folder/(name+'_tokens.json'),seq);pp.append(seq)
    result={'same_preprocessing_token_sequence':pp[0]==pp[1],
        'original_token_sha256':sha(folder/'original_tokens.json'),'view_token_sha256':sha(folder/'view_tokens.json'),
        'compiler_path':compiler,'compiler_sha256':sha(compiler),'Frama_generated_macro_file_hashes':macro_files,
        'compiler_profile_flags':[a for a in argv if a.startswith(('-std=','-W'))],
        'cpp_recipe_source':'actual same installed Frama-C -print-cpp-commands',
        'only_command_path_substitutions':['source input path','preprocessed output path'],
        'token_exclusions':['cpp line-location metadata only','ordinary non-ACSL comments'],
        'macro_definition_undef_and_header_pragma_records':'retained exactly as additional tokens',
        'native_profile_equivalent_for_preprocessing':options}
    save(folder/'token_equivalence.json',result)
    if not result['same_preprocessing_token_sequence']:raise ValueError('PREPROCESSED_TOKEN_MISMATCH')
    return result


def response(r):return r.get('data') if isinstance(r,dict) and r.get('res')=='DATA' and not r.get('error') else None


def identity_rows(source,unit_id,entry,mapping,raw):
    originals,facts=source_inventory(source,unit_id,entry)
    source_by_span={(t['source_span']['start_byte'],t['source_span']['end_byte'],t['role']):t for t in originals}
    ts=raw.get('source_targets',[]);native=raw.get('native_exports',{})
    probes=[response(t.get('probe_info')) or {} for t in ts]
    stmt_counts={p.get('stmt'):sum(q.get('stmt')==p.get('stmt') for q in probes) for p in probes if p.get('stmt')}
    attrs={a['marker']:a for a in (response(native.get('marker_attributes')) or {}).get('updated',[])}
    rows=[]
    for t,p in zip(ts,probes):
        span=t['source_span'];start=original_offset(mapping,span['start_byte']);end=original_offset(mapping,span['end_byte'],True)
        orig=source_by_span.get((start,end,t['role']));reasons=[];stmt=p.get('stmt')
        if orig is None:reasons.append('NO_ORIGINAL_ACCESS_AT_INVERSE_VIEW_SPAN')
        if not stmt:reasons.append('NO_ACTUAL_PROBE_KINSTR')
        if stmt and stmt_counts[stmt]!=1:reasons.append('MULTIPLE_VIEW_TARGETS_SHARE_KINSTR')
        if t['kind']!='array_index':reasons.append('LIBRARY_ROLE_BINDING_UNSUPPORTED')
        if t.get('unevaluated') or t.get('formation'):reasons.append('NOT_EVALUATED_ACCESS')
        if p.get('code')!=t['source_text']:reasons.append('NATIVE_PROBE_CODE_DIFFERS_FROM_VIEW_ACCESS')
        loc=attrs.get(response(t.get('marker')),{}).get('sloc',{})
        if loc.get('file') not in raw['request']['command'] or loc.get('line')!=span['start_line']:
            reasons.append('NATIVE_VIEW_SOURCE_LOCATION_NOT_CONFIRMED')
        if not response(t.get('parsed_operand_marker')):reasons.append('NO_OPERAND_PARSED_AT_ACTUAL_KINSTR')
        contexts=response(t.get('callstack_ids')) or [];values=t.get('access_point_values',[])
        if not contexts or len(contexts)!=len(values) or any(not (response(v.get('evaluation')) or {}).get('vBefore') for v in values):
            reasons.append('ACTUAL_CONTEXT_BEFORE_STATE_UNAVAILABLE')
        rows.append({'source_access_id':orig['source_access_id'] if orig else None,
            'original_source_span':orig['source_span'] if orig else None,'view_source_span':span,
            'role':t['role'],'source_text':t['source_text'],'native_marker':response(t.get('marker')),
            'native_probe_kinstr':stmt,'native_operand_marker':response(t.get('parsed_operand_marker')),
            'callstack_ids':contexts,'identity_state':'BOUND_SINGLE_ACCESS' if not reasons else 'UNKNOWN',
            'unknown_reasons':reasons,'classification_verdict':'UNKNOWN'})
    seen={r['source_access_id'] for r in rows}
    for orig in originals:
        if orig['source_access_id'] not in seen:
            rows.append({'source_access_id':orig['source_access_id'],'original_source_span':orig['source_span'],
                'identity_state':'UNKNOWN','unknown_reasons':['NO_VIEW_TARGET_FOR_ORIGINAL_ACCESS'],'classification_verdict':'UNKNOWN'})
    return rows,facts


def export_identity(source,unit_id,folder,entry='main'):
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    identity={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'unit_id':unit_id,'entry':entry,
        'implementation_sha256':sha(__file__),'exporter_sha256':sha(Path(__file__).with_name('binding_probe_export.py')),
        'cpp_capture_sha256':sha(Path(__file__).with_name('binding_cpp_capture.py')),
        'tool_sha256':sha(TOOL),'machdep_sha256':sha(SHARE/'machdeps/machdep_x86_64.yaml'),
        'parent_dependency_hashes':{name:sha(Path(__file__).with_name(name)) for name in
            ['evidence_access.py','evidence_numeric.py','evidence_char_extension.py']},
        'ABI_file_sha256':sha(OLD/'numeric_acceptance/abi.json'),
        'encoding_contract_sha256':sha(CHAR/'character_contract.json'),
        'parser_versions':{n:importlib.metadata.version(n) for n in ['tree-sitter','tree-sitter-c']},
        'environment_sha256':hashlib.sha256(json.dumps(dict(os.environ),sort_keys=True).encode()).hexdigest(),
        'scope':'whitespace-view-identity-only-v1'}
    completion=folder/'completion.json'
    if completion.exists():
        old=json.loads(completion.read_text())
        if old['identity']!=identity:raise RuntimeError('CACHE_IDENTITY_MISMATCH_USE_NEW_ATTEMPT')
        for name,h in old['artifacts'].items():
            if sha(folder/name)!=h:raise RuntimeError('CACHE_ARTIFACT_MISMATCH:'+name)
        for name,h in old.get('external_dependencies',{}).items():
            if sha(name)!=h:raise RuntimeError('CACHE_EXTERNAL_DEPENDENCY_MISMATCH:'+name)
        return json.loads((folder/'result.json').read_text())
    if any(folder.iterdir()):raise RuntimeError('NONEMPTY_ATTEMPT_REFUSE_OVERWRITE')
    save(folder/'request.json',identity);(folder/'original.c').write_text(source)
    result={'schema':'binding_identity_v1','identity':identity,'targets':[],'failure':None,
        'identity_gate':'UNKNOWN','classification_gate':'UNKNOWN','full_C_program_safety':'UNKNOWN',
        'dataset_program_execution':False,'facts_used_for_classification':False}
    try:
        original_targets,original_facts=source_inventory(source,unit_id,entry)
        result['source_inventory']=original_facts
        result['targets']=[{'source_access_id':t['source_access_id'],'original_source_span':t['source_span'],
            'identity_state':'UNKNOWN','unknown_reasons':['IDENTITY_PIPELINE_NOT_COMPLETED'],
            'classification_verdict':'UNKNOWN'} for t in original_targets]
        view,mapping=make_view(source);save(folder/'position_mapping.json',mapping)
        (folder/'view.c').write_text(view)
        token_info=cpp_equivalence(folder/'original.c',folder/'view.c',folder,entry)
        capture=folder/'native_cpp_capture';capture.mkdir()
        updates={'CPP':shlex.join([sys.executable,str(Path(__file__).with_name('binding_cpp_capture.py')),*token_info['compiler_profile_flags']]),
            'BINDING_CPP_CAPTURE_DIR':str(capture),'BINDING_CPP_COMPILER':token_info['compiler_path']}
        previous={k:os.environ.get(k) for k in updates}
        try:
            os.environ.update(updates)
            raw=export_fixture(view,unit_id+':view',folder/'native_view',entry)
        finally:
            for k,v in previous.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v
        if raw['failure'] or raw['exit_code']!=0:raise RuntimeError('NATIVE_VIEW_EXPORT_FAILED')
        candidates=[]
        for f in capture.glob('*/receipt.json'):
            receipt=json.loads(f.read_text())
            if str(folder/'native_view/input.c') in receipt['command']:candidates.append((f,receipt))
        if len(candidates)!=1:raise ValueError('ACTUAL_NATIVE_CPP_CAPTURE_NOT_UNIQUE')
        capture_file,capture_receipt=candidates[0]
        native_tokens=tokens((capture_file.parent/'output.i').read_bytes());save(folder/'native_view_tokens.json',native_tokens)
        if native_tokens!=json.loads((folder/'view_tokens.json').read_text()):raise ValueError('ACTUAL_NATIVE_CPP_TOKEN_MISMATCH')
        save(folder/'actual_native_token_equivalence.json',{'same_tokens':True,'capture_receipt':str(capture_file.relative_to(folder)),
            'native_token_sha256':sha(folder/'native_view_tokens.json'),'view_token_sha256':sha(folder/'view_tokens.json'),
            'capture_configuration':updates,'compiler_sha256':capture_receipt['compiler_sha256']})
        # Compare observed actual native -H headers against the shared original
        # and view recipe, and re-hash them after native analysis.
        headers=[]
        for name in ['original','view']:
            paths=set()
            for line in (folder/('cpp_'+name+'.stderr')).read_text().splitlines():
                if line.startswith('.'):
                    p=Path(line.lstrip('.').strip())
                    if p.is_file():paths.add(str(p.resolve()))
            headers.append({('FRAMA_GENERATED:'+Path(p).name if '/tmp/__fc_machdep' in p else p):sha(p) for p in paths})
        native_headers={('FRAMA_GENERATED:'+Path(h['path']).name if '/tmp/__fc_machdep' in h['path'] else h['path']):h['sha256'] for h in capture_receipt['headers']}
        if not(headers[0]==headers[1]==native_headers):raise ValueError('ACTUAL_HEADER_CLOSURE_MISMATCH')
        save(folder/'actual_header_equivalence.json',{'original':headers[0],'view':headers[1],'native_view':native_headers,'equal':True})
        rows,facts=identity_rows(source,unit_id,entry,mapping,raw);result['targets']=rows
        result['source_inventory']=facts
        result['identity_gate']='BOUND_SINGLE_ACCESS' if rows and all(r['identity_state']=='BOUND_SINGLE_ACCESS' for r in rows) else 'UNKNOWN'
        result['native_view_result']='native_view/export.json'
    except Exception as exc:
        result['failure']=type(exc).__name__+':'+str(exc)
    save(folder/'result.json',result)
    artifacts={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='completion.json'}
    # Nested native completion is identity evidence too, not silently excluded.
    for p in folder.rglob('completion.json'):
        if p!=completion:artifacts[str(p.relative_to(folder))]=sha(p)
    dependencies={}
    header_file=folder/'actual_header_equivalence.json'
    if header_file.exists():dependencies.update({k:v for k,v in json.loads(header_file.read_text())['native_view'].items() if not k.startswith('FRAMA_GENERATED:')})
    token_file=folder/'token_equivalence.json'
    if token_file.exists():
        t=json.loads(token_file.read_text());dependencies[t['compiler_path']]=t['compiler_sha256']
    save(completion,{'identity':identity,'artifacts':artifacts,'external_dependencies':dependencies})
    return result

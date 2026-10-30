"""Strict original-query to evaluated source-syntax traceability, not a classifier."""
import hashlib
from tree_sitter import Language,Parser
import tree_sitter_c
SCHEMA='query_access_linkage_v1'
CALL_ROLES={'memset':['copy_destination'],'memcpy':['copy_destination','copy_source'],'memmove':['copy_destination','copy_source']}
MEMORY_NAMES=set(CALL_ROLES)|{'strcpy','strncpy','strcat','strncat','sprintf','snprintf','vsprintf','vsnprintf','gets','fgets','scanf','fscanf','sscanf'}
def walk(n):
    todo=[n]
    while todo:
        a=todo.pop();yield a;todo.extend(reversed(a.named_children))
def text(raw,n):return raw[n.start_byte:n.end_byte].decode('utf-8') if n else ''
def span(n):return {'start_byte':n.start_byte,'end_byte':n.end_byte,'start_line':n.start_point.row+1,'end_line':n.end_point.row+1,'start_column':n.start_point.column,'end_column':n.end_point.column}
def function_name(raw,n):
    d=n.child_by_field_name('declarator')
    while d and d.type!='identifier':d=d.child_by_field_name('declarator')
    return text(raw,d)
def context(n,raw):
    reasons=[];p=n.parent
    while p and p.type!='function_definition':
        if p.type in ['sizeof_expression','alignof_expression','generic_expression','type_descriptor']:reasons.append('UNEVALUATED_OR_TYPE_CONTEXT')
        if p.type=='pointer_expression' and text(raw,p.child_by_field_name('operator'))=='&':reasons.append('ADDRESS_FORMATION_CONTEXT')
        p=p.parent
    return sorted(set(reasons))
def role(n,raw):
    p=n.parent
    while p and p.type=='parenthesized_expression':p=p.parent
    if p and p.type=='update_expression':return 'read_write'
    if p and p.type=='assignment_expression':
        left=p.child_by_field_name('left')
        if left and left.start_byte<=n.start_byte and n.end_byte<=left.end_byte:
            return 'write' if text(raw,p.child_by_field_name('operator'))=='=' else 'read_write'
    return 'read'
def link_query(source,registration,source_identity):
    raw=source.encode('utf-8');digest=hashlib.sha256(raw).hexdigest();sid=registration['scene_id'];line=registration['query_line'];entry=source_identity.get('entry','main')
    out={'schema_version':SCHEMA,'unit_id':sid,'source_case':registration['source_case'],'source_sha256':digest,'entry':entry,'query_line':line,'row':registration.get('row'),'family_id':registration.get('family_id'),'source_access_id':None,'candidates':[],'linkage_status':'UNKNOWN','reasons':[],'native_binding':'NOT_ATTEMPTED','complete_obligations':False,'classification':'UNKNOWN','full_C_program_safety':'UNKNOWN','indirect_defuse_enabled':False}
    if digest!=source_identity['source_sha256']:out['reasons']=['SOURCE_HASH_MISMATCH'];return out
    if type(line) is not int or line<1 or line>len(source.splitlines()):out['reasons']=['QUERY_LINE_INVALID'];return out
    parser=Parser(Language(tree_sitter_c.language()));tree=parser.parse(raw);out['whole_translation_unit_has_diagnostics']=tree.root_node.has_error
    funcs=[f for f in tree.root_node.named_children if f.type=='function_definition' and function_name(raw,f)==entry]
    if len(funcs)!=1:out['reasons']=['ENTRY_NOT_UNIQUE'];return out
    f=funcs[0];out['entry_function_AST_valid']=not f.has_error
    if f.has_error:out['reasons']=['ENTRY_FUNCTION_AST_ERROR'];return out
    if not f.start_point.row+1<=line<=f.end_point.row+1:out['reasons']=['QUERY_NOT_IN_ENTRY'];return out
    # Macro-generated, unexpanded or implicit accesses are not reconstructed.
    for n in walk(f.child_by_field_name('body')):
        if not n.start_point.row+1<=line<=n.end_point.row+1:continue
        kind=None;roles=[];extra=[]
        if n.type=='subscript_expression':kind='array_index';roles=[role(n,raw)]
        elif n.type=='pointer_expression' and text(raw,n.child_by_field_name('operator'))=='*':kind='pointer_dereference';roles=[role(n,raw)]
        elif n.type=='field_expression' and text(raw,n.child_by_field_name('operator'))=='->':kind='indirect_field_access';roles=[role(n,raw)];extra=['FIELD_TYPE_LAYOUT_NOT_CERTIFIED']
        elif n.type=='call_expression':
            callee=text(raw,n.child_by_field_name('function'))
            if callee not in MEMORY_NAMES:continue
            kind='memory_call';roles=CALL_ROLES.get(callee,['unresolved_memory_call_role']);extra=['CALLEE_IDENTITY_AND_CONTRACT_NOT_CERTIFIED']
            if callee not in CALL_ROLES:extra.append('MEMORY_CALL_ROLE_UNSUPPORTED')
            defs=[d for d in tree.root_node.named_children if d.type=='function_definition' and function_name(raw,d)==callee]
            if defs:extra.append('CUSTOM_SAME_NAME_CALL_UNTRUSTED')
        if not kind:continue
        excluded=context(n,raw);out['candidates'].append({'source_span':span(n),'source_text':text(raw,n),'kind':kind,'roles':roles,'source_access_ids':[sid+':'+str(n.start_byte)+':'+str(n.end_byte)+((':'+r) if kind=='memory_call' else '') for r in roles],'excluded_reasons':excluded,'semantic_caveats':extra,'evaluated_syntax_candidate':not excluded})
    eligible=[a for a in out['candidates'] if not a['excluded_reasons']]
    if not eligible:out['reasons']=['NO_EVALUATED_MEMORY_ACCESS_ON_QUERY_LINE','NO_ORIGINAL_DDG_CERTIFICATE'];return out
    if len(eligible)!=1:out['reasons']=['MULTIPLE_OR_NESTED_MEMORY_ACCESS_CANDIDATES'];return out
    a=eligible[0]
    if len(a['roles'])!=1 or a['kind'] in ['memory_call','indirect_field_access']:out['reasons']=['MEMORY_CALL_ROLE_OR_CALLEE_NOT_CERTIFIED'];return out
    out['linkage_status']='DIRECT_SYNTAX_UNIQUE';out['source_access_id']=a['source_access_ids'][0];out['unique_access']=a
    out['reasons']=['SOURCE_SYNTAX_ONLY_NATIVE_IDENTITY_AND_PRECONDITIONS_NOT_CERTIFIED'];return out

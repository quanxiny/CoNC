"""Finite source-address diagnostic, separate from frozen X/E/N and identity.

Only potential address-bound violations or UNKNOWN. Memory contents remain
unknown. A modeled trace is not a defined C execution or reachable bug proof.
"""
import copy
import hashlib
import json
from pathlib import Path
import tree_sitter
import tree_sitter_c
from evidence_char_extension import CharacterAnalyzer
from evidence_numeric import CType, typed_op, convert, check_bounds
from binding_probe_export import OLD, CHAR, sha
from binding_identity import nodes, source_tree_acceptable

_LANGUAGE=tree_sitter.Language(tree_sitter_c.language())
_PARSER=tree_sitter.Parser(_LANGUAGE)
def parse(raw):return _PARSER.parse(raw)


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


class Stop(Exception):pass
class BreakLoop(Exception):pass
class ReturnEntry(Exception):pass


class AddressDomain(CharacterAnalyzer):
    def __init__(self,source,unit_id,query_span,entry='main',library_binding=None,budget=512):
        abi=json.loads((OLD/'numeric_acceptance/abi.json').read_text())
        super().__init__(source,unit_id,'',abi,entry,None,{'character_contract':json.loads((CHAR/'character_contract.json').read_text())})
        self.target_span=(query_span['start_byte'],query_span['end_byte']);self.budget=budget;self.steps=0
        self.trace=[];self.reads=[];self.assumptions=[];self.unknown=[];self.witness=None;self.active={};self.binding=library_binding
        self.lib_ok=False
        if library_binding:
            b={k:v for k,v in library_binding.items() if k!='sha256'}
            self.lib_ok=library_binding.get('sha256')==digest(b) and b.get('source_sha256')==hashlib.sha256(self.src).hexdigest()
            try:
                self.lib_ok &= sha(b['source_CPP_file'])==b['source_CPP_sha256'] and sha(b['compiler'])==b['compiler_sha256']
                self.lib_ok &= all(sha(p)==h for p,h in b['headers'].items())
            except (KeyError,OSError):self.lib_ok=False

    def stop(self,why):raise Stop(why)
    def tick(self,n,what):
        self.steps+=1
        if self.steps>self.budget:self.stop('STEP_BUDGET_EXHAUSTED')
        if len(self.trace)<self.budget:self.trace.append({'step':self.steps,'span':self.span(n),'kind':what})
    def scalar_type(self,typ):
        ct=self.ct(typ)
        if ct is None or not ct.supported:self.stop('UNSUPPORTED_INTEGER_TYPE')
        return ct
    def descriptor(self,td):
        if td is None or td.type!='type_descriptor':self.stop('TYPE_DESCRIPTOR_MISSING')
        if any(n.type=='type_qualifier' for n in nodes(td)):self.stop('TYPE_DESCRIPTOR_QUALIFIER_UNSUPPORTED')
        base=self.base_type(self.field(td,'type'))
        if base.get('kind')!='integer' or self.size(base) is None:self.stop('TYPE_DESCRIPTOR_BASE_UNSUPPORTED')
        return base,self.field(td,'declarator')
    def simple_pointer_descriptor(self,td):
        base,dec=self.descriptor(td)
        if dec is None or dec.type!='abstract_pointer_declarator' or dec.named_children:
            self.stop('POINTER_DESCRIPTOR_DEPTH_OR_DERIVATION_UNSUPPORTED')
        return base
    def cast_scalar(self,v,st,tt):
        r=convert(v,st,tt)
        if v is not None and r['value'] is None:self.stop('CONVERSION:'+str(r['reason']))
        return r['value'],tt
    def operation(self,op,a,b,ta,tb):
        r=typed_op(op,a,b,ta,tb)
        if r['reason']=='signed_overflow':self.stop('KNOWN_SIGNED_OVERFLOW')
        if r['type']:return r['value'],CType(**r['type'])
        if a is None or b is None:
            if ta and tb and ta.supported and tb.supported:
                t=typed_op(op,0,0,ta,tb)['type'];return None,CType(**t) if t else None
        self.stop('UNSUPPORTED_TYPED_ARITHMETIC:'+str(r['reason']))
    def pointer_value(self,n):
        value,_=self.eval(n)
        return value if isinstance(value,dict) else None
    def pointer_shift(self,p,index,formation=True):
        if p is None or not isinstance(index,int):return None
        size=self.size(p['pointee'])
        if size is None:return None
        q=copy.deepcopy(p);q['offset']+=index*size
        q['index_evaluation']={'value':index,'element_bytes':size,'address_arithmetic':'mathematical object-relative bytes; C index evaluation separate'}
        if formation and check_bounds(q['offset'],0,q['capacity'],formation=True)['relation']!='BOUND_SATISFIED':
            self.stop('PRIOR_INVALID_POINTER_FORMATION')
        return q
    def location_at(self,n):
        if n.type=='subscript_expression':
            p=self.pointer_value(self.field(n,'argument'));i,_=self.eval(self.field(n,'index'))
            return self.pointer_shift(p,i,False)
        if n.type=='pointer_expression' and self.text(self.field(n,'operator'))=='*':
            arg=self.unwrap(self.field(n,'argument'))
            if arg.type=='binary_expression' and self.text(self.field(arg,'operator')) in ('+','-'):
                p=self.pointer_value(self.field(arg,'left'));i,_=self.eval(self.field(arg,'right'))
                return self.pointer_shift(p,i if self.text(self.field(arg,'operator'))=='+' else -i if isinstance(i,int) else None,False)
            return self.pointer_value(arg)
        return None
    def access(self,n,p,mode):
        is_query=(n.start_byte,n.end_byte)==self.target_span
        if not is_query:
            if p is None or not self.active.get(p['object_id']):self.stop('NONQUERY_ADDRESS_OR_LIFETIME_UNKNOWN')
            if check_bounds(p['offset'],self.size(p['pointee']),p['capacity'])['relation']!='BOUND_SATISFIED':
                self.stop('PRIOR_NONQUERY_BOUNDS_NOT_SATISFIED')
        if is_query:
            self.tick(n,'query_address_evaluation')
            if p is None:self.stop('QUERY_OBJECT_OR_INDEX_UNKNOWN')
            if not self.active.get(p['object_id']):self.stop('QUERY_OBJECT_LIFETIME_NOT_ACTIVE')
            width=self.size(p['pointee'])
            if width is None:self.stop('QUERY_POINTEE_WIDTH_UNKNOWN')
            relation=check_bounds(p['offset'],width,p['capacity'])
            if relation['relation']=='BOUND_VIOLATED':
                self.witness={'source_span':self.span(n),'source_text':self.text(n),'object_id':p['object_id'],
                    'offset_bytes':p['offset'],'width_bytes':width,'capacity_bytes':p['capacity'],'unit':'bytes',
                    'pointee_type':p['pointee'],'object_provenance':p['provenance'],'access_mode':mode,
                    'relation':'BOUND_VIOLATED','trace_step':self.steps,'index_evaluation':p.get('index_evaluation')}
        if mode=='read':self.reads.append({'span':self.span(n),'content':'UNKNOWN','used_value_not_invented':True})
        return None,self.ct(p['pointee']) if p else None
    def condition(self,n):
        v,_=self.eval(n)
        if not isinstance(v,(int,dict)):self.stop('CONTROL_DEPENDS_ON_UNKNOWN_VALUE')
        return bool(v)

    def names_and_writes(self,n):
        names=set();writes=set()
        for c in nodes(n):
            if c.type=='identifier':names.add(self.text(c))
            if c.type in ('update_expression','assignment_expression'):
                lhs=self.field(c,'argument') if c.type=='update_expression' else self.field(c,'left')
                if lhs is not None:
                    writes.update(self.text(x) for x in nodes(lhs) if x.type=='identifier')
        return names,writes

    def eval(self,node):
        n=self.unwrap(node)
        if n is None:return None,None
        self.tick(n,'expression:'+n.type)
        if n.type in ('number_literal','char_literal'):
            value,ct=CharacterAnalyzer.number(self,n)
            if value is None:self.stop('LITERAL_UNSUPPORTED')
            return value,ct
        if n.type=='identifier':
            name=self.text(n)
            if name=='NULL' and self.lib_ok:return 0,CType(32,True)
            v=self.env.get(name)
            if v is None:self.stop('IDENTIFIER_NOT_LOCAL_OR_VERIFIED:'+name)
            return copy.deepcopy(v['pointer']) if v['type']['kind'] in ('array','pointer') else v['value'],self.ct(v['type'])
        if n.type=='sizeof_expression':
            arg=self.field(n,'value');typ=self.field(n,'type')
            if arg is not None and self.unwrap(arg).type=='identifier':t=self.env.get(self.text(self.unwrap(arg)),{}).get('type')
            elif typ is not None:
                t,dec=self.descriptor(typ)
                if dec is not None:self.stop('SIZEOF_DERIVED_TYPE_UNSUPPORTED')
            else:self.stop('SIZEOF_UNSUPPORTED')
            size=self.size(t)
            if size is None:self.stop('SIZEOF_TYPE_UNKNOWN')
            return size,CType(64,False)
        if n.type=='cast_expression':
            td=self.field(n,'type');base,decl=self.descriptor(td)
            if decl is not None:
                pointee=self.simple_pointer_descriptor(td);value,_=self.eval(self.field(n,'value'))
                if not isinstance(value,dict) or value['pointee']!=pointee:
                    self.stop('POINTER_CAST_POINTEE_MISMATCH_OR_UNKNOWN')
                return value,None
            v,st=self.eval(self.field(n,'value'));return self.cast_scalar(v,st,self.scalar_type(base))
        if n.type=='update_expression':
            arg=self.field(n,'argument')
            if arg.type!='identifier':self.stop('UPDATE_TARGET_UNSUPPORTED')
            name=self.text(arg);old=self.env.get(name)
            if old is None:self.stop('UPDATE_UNKNOWN_SCALAR')
            ct=self.scalar_type(old['type']);value=old['value'];op=self.text(self.field(n,'operator'))
            nv,nt=self.operation('ADD' if op=='++' else 'SUB',value,1,ct,CType(32,True))
            old['value'],_=self.cast_scalar(nv,nt,ct)
            return (old['value'] if n.children[0].type in ('++','--') else value),ct
        if n.type=='unary_expression':
            op=self.text(self.field(n,'operator'));v,t=self.eval(self.field(n,'argument'))
            if op=='-':return self.operation('SUB',0,v,CType(32,True),t)
            if op=='!':return (int(not v) if isinstance(v,(int,dict)) else None),CType(32,True)
            self.stop('UNARY_OPERATOR_UNSUPPORTED')
        if n.type=='binary_expression':
            op=self.text(self.field(n,'operator'))
            if op not in ('&&','||'):
                ln,lw=self.names_and_writes(self.field(n,'left'));rn,rw=self.names_and_writes(self.field(n,'right'))
                if (lw & rn) or (rw & ln):self.stop('UNSEQUENCED_CONFLICTING_OPERAND_EFFECTS')
            a,ta=self.eval(self.field(n,'left'))
            if op in ('&&','||'):
                if a is None:return None,CType(32,True)
                if (op=='&&' and not a) or (op=='||' and a):return int(bool(a)),CType(32,True)
                b,_=self.eval(self.field(n,'right'));return (int(bool(b)) if b is not None else None),CType(32,True)
            b,tb=self.eval(self.field(n,'right'))
            if op in ('+','-'):
                if isinstance(a,dict):return self.pointer_shift(a,b if op=='+' else -b if isinstance(b,int) else None),None
                return self.operation('ADD' if op=='+' else 'SUB',a,b,ta,tb)
            if op in ('<','<=','>','>=','==','!='):
                if a is None or b is None:return None,CType(32,True)
                if isinstance(a,dict) or isinstance(b,dict):
                    if op not in ('==','!=') or not ((isinstance(a,dict) and b==0) or (isinstance(b,dict) and a==0)):
                        self.stop('POINTER_COMPARISON_UNSUPPORTED')
                    return int(op=='!='),CType(32,True)
                common=typed_op('ADD',0,0,ta,tb).get('type')
                if common is None:self.stop('COMPARISON_TYPE_UNSUPPORTED')
                ct=CType(**common);aa=convert(a,ta,ct)['value'];bb=convert(b,tb,ct)['value']
                return int({'<':aa<bb,'<=':aa<=bb,'>':aa>bb,'>=':aa>=bb,'==':aa==bb,'!=':aa!=bb}[op]),CType(32,True)
            self.stop('BINARY_OPERATOR_UNSUPPORTED:'+op)
        if n.type=='conditional_expression':
            self.stop('CONDITIONAL_COMMON_TYPE_UNSUPPORTED')
        if n.type=='assignment_expression':
            lhs=self.field(n,'left');rhs=self.field(n,'right');op=self.text(self.field(n,'operator'))
            ln,lw=self.names_and_writes(lhs);rn,rw=self.names_and_writes(rhs)
            if (lhs.type=='identifier' and self.text(lhs) in rw) or (lhs.type!='identifier' and ((lw & rn) or (rw & ln))):
                self.stop('UNSEQUENCED_ASSIGNMENT_EFFECTS')
            if lhs.type!='identifier':
                if op!='=':self.stop('MEMORY_COMPOUND_ASSIGNMENT_UNSUPPORTED')
                if any(c.type in ('update_expression','assignment_expression','call_expression') for c in nodes(rhs)):
                    self.stop('MEMORY_WRITE_RHS_SIDE_EFFECTS_UNSUPPORTED')
                self.eval(rhs);p=self.location_at(lhs);self.access(lhs,p,'write');return None,None
            name=self.text(lhs);v=self.env.get(name)
            if v is None:self.stop('ASSIGNMENT_TARGET_UNKNOWN')
            value,st=self.eval(rhs)
            if v['type']['kind']=='pointer':
                if op!='=':self.stop('POINTER_UPDATE_UNSUPPORTED')
                if value is not None and (not isinstance(value,dict) or value['pointee']!=v['type']['element']):self.stop('POINTER_TYPE_MISMATCH')
                v['pointer']=value;return value,None
            ct=self.scalar_type(v['type'])
            if op in ('+=','-='):value,st=self.operation('ADD' if op=='+=' else 'SUB',v['value'],value,ct,st)
            elif op!='=':self.stop('ASSIGNMENT_OPERATOR_UNSUPPORTED')
            v['value'],_=self.cast_scalar(value,st,ct);return v['value'],ct
        if n.type=='subscript_expression' or (n.type=='pointer_expression' and self.text(self.field(n,'operator'))=='*'):
            return self.access(n,self.location_at(n),'read')
        if n.type=='pointer_expression' and self.text(self.field(n,'operator'))=='&':
            arg=self.unwrap(self.field(n,'argument'))
            if arg.type!='subscript_expression':self.stop('ADDRESS_OF_SCALAR_UNSUPPORTED')
            p=self.location_at(arg)
            if p is None:self.stop('ADDRESS_FORMATION_UNKNOWN')
            if check_bounds(p['offset'],0,p['capacity'],formation=True)['relation']!='BOUND_SATISFIED':self.stop('PRIOR_INVALID_POINTER_FORMATION')
            return p,None
        if n.type=='call_expression':
            fn=self.text(self.field(n,'function'));args=self.field(n,'arguments').named_children
            if not self.lib_ok or fn in self.env:self.stop('UNVERIFIED_CALL:'+fn)
            if fn=='assert' and self.binding.get('assert_macro'):
                if len(args)!=1 or not self.condition(args[0]):self.stop('ASSERT_TERMINATES_OR_UNKNOWN')
                return None,None
            if fn=='free' and len(args)==1:
                p=self.pointer_value(args[0])
                if p is None or p['offset']!=0 or ':heap:' not in p['object_id'] or not self.active.get(p['object_id']):self.stop('FREE_TARGET_UNKNOWN')
                self.active[p['object_id']]=False;return None,None
            self.stop('UNSUPPORTED_VERIFIED_CALL:'+fn)
        self.stop('EXPRESSION_UNSUPPORTED:'+n.type)

    def allocation(self,n,element):
        value=self.unwrap(n)
        if value and value.type=='cast_expression':
            td=self.field(value,'type')
            if self.simple_pointer_descriptor(td)!=element:
                self.stop('MALLOC_CAST_POINTEE_NOT_VERIFIED')
            value=self.unwrap(self.field(value,'value'))
        if value is None or value.type!='call_expression' or self.text(self.field(value,'function'))!='malloc':return None
        if not self.lib_ok or 'malloc' in self.env:self.stop('MALLOC_DECLARATION_OR_HEADER_NOT_BOUND')
        args=self.field(value,'arguments').named_children
        if len(args)!=1:self.stop('MALLOC_ARGUMENTS_UNSUPPORTED')
        expr=self.unwrap(args[0])
        if expr.type!='binary_expression' or self.text(self.field(expr,'operator'))!='*':self.stop('MALLOC_EXTENT_PATTERN_UNSUPPORTED')
        count=self.unwrap(self.field(expr,'left'));sz=self.unwrap(self.field(expr,'right'))
        if count.type!='number_literal' or sz.type!='sizeof_expression':self.stop('MALLOC_EXTENT_PATTERN_UNSUPPORTED')
        cv,ct=self.eval(count);typ=self.field(sz,'type')
        et,dec=self.descriptor(typ)
        if dec is not None:
            self.stop('MALLOC_SIZEOF_DERIVED_TYPE_UNSUPPORTED')
        width=self.size(element)
        if et!=element or not isinstance(cv,int) or cv<=0 or width is None or cv>((1<<64)-1)//width:
            self.stop('MALLOC_SIZEOF_TYPE_OR_SIZE_T_PRODUCT_UNVERIFIED')
        if self.mac.get('__SIZEOF_POINTER__')!='8' or self.mac.get('__SIZEOF_LONG__')!='8':self.stop('SIZE_T_ABI_NOT_VERIFIED')
        oid=self.uid+':heap:'+str(n.start_byte)
        if oid in self.active:self.stop('REPEATED_ALLOCATION_SITE_UNSUPPORTED')
        self.active[oid]=True
        self.assumptions.append({'kind':'malloc_allocation_succeeds','span':self.span(value),'not_proved':True})
        return {'object_id':oid,'offset':0,'capacity':cv*width,'pointee':element,
            'provenance':[{'kind':'verified_standard_malloc_literal_count_sizeof_pointee','span':self.span(value),
                'count':cv,'element_bytes':width,'size_t_bits':64,'product_range_checked':True,'library_binding_sha256':self.binding['sha256']}]}

    def declare(self,n):
        if any(c.type in ('type_qualifier','storage_class_specifier') for c in nodes(n)):self.stop('QUALIFIED_OR_STORAGE_DECLARATION_UNSUPPORTED')
        base=self.base_type(self.field(n,'type'))
        for i,d in enumerate(n.children):
            if n.field_name_for_child(i)!='declarator':continue
            init=self.field(d,'value') if d.type=='init_declarator' else None
            dec=self.field(d,'declarator') if d.type=='init_declarator' else d
            name,typ=self.declarator(dec,base)
            if not name or name in self.env:self.stop('DECLARATION_SHADOW_OR_TYPE_UNSUPPORTED')
            item={'type':typ,'value':None,'pointer':None}
            if typ['kind']=='array':
                if init is not None and any(c.type in ('update_expression','assignment_expression','call_expression') for c in nodes(init)):
                    self.stop('ARRAY_INITIALIZER_STATE_EFFECTS_UNSUPPORTED')
                size=self.size(typ)
                if size is None or size<=0 or typ['element'].get('kind')!='integer':self.stop('ARRAY_TYPE_OR_EXTENT_UNSUPPORTED')
                oid=self.uid+':stack:'+str(dec.start_byte)
                if oid in self.active:self.stop('REPEATED_STACK_LIFETIME_UNSUPPORTED')
                self.active[oid]=True
                item['pointer']={'object_id':oid,'offset':0,'capacity':size,'pointee':typ['element'],
                    'provenance':[{'kind':'fixed_local_array','span':self.span(n),'ABI_sha256':self.abi['sha256']}]}
                # Even initialized contents are not interpreted in this domain.
                if init is not None and init.type not in ('initializer_list','string_literal'):self.stop('ARRAY_INITIALIZER_UNSUPPORTED')
            elif typ['kind']=='pointer':
                item['pointer']=self.allocation(init,typ['element']) if init is not None else None
                if init is not None and item['pointer'] is None:
                    p=self.pointer_value(init)
                    if p is not None and p['pointee']!=typ['element']:self.stop('POINTER_TYPE_MISMATCH')
                    item['pointer']=p
            elif typ['kind']=='integer':
                ct=self.scalar_type(typ)
                if init is not None:
                    v,st=self.eval(init);item['value'],_=self.cast_scalar(v,st,ct)
            else:self.stop('DECLARATION_TYPE_UNSUPPORTED')
            self.env[name]=item

    def statement(self,n):
        self.tick(n,'statement:'+n.type)
        if n.type=='compound_statement':
            before=set(self.env);before_objects=set(self.active)
            try:
                for child in n.named_children:
                    if child.type!='comment':self.statement(child)
            finally:
                for key in set(self.env)-before:self.env.pop(key,None)
                for oid in set(self.active)-before_objects:
                    if ':stack:' in oid:self.active[oid]=False
        elif n.type=='declaration':self.declare(n)
        elif n.type=='expression_statement':
            if n.named_children:self.eval(n.named_children[0])
        elif n.type=='return_statement':
            if n.named_children:self.eval(n.named_children[0])
            if self.witness is not None:self.stop('ADDRESS_POTENTIAL_WITNESS_FOUND')
            raise ReturnEntry()
        elif n.type=='if_statement':
            branch=self.field(n,'consequence') if self.condition(self.field(n,'condition')) else self.field(n,'alternative')
            if branch is not None:self.statement(branch.named_children[0] if branch.type=='else_clause' else branch)
        elif n.type in ('while_statement','for_statement'):
            if n.type=='for_statement':
                ini=self.field(n,'initializer')
                if ini is not None and any(c.type=='array_declarator' for c in nodes(ini)):
                    self.stop('FOR_INITIALIZER_ARRAY_LIFETIME_UNSUPPORTED')
                if ini is not None:self.statement(ini) if ini.type=='declaration' else self.eval(ini)
            while self.condition(self.field(n,'condition')) if self.field(n,'condition') is not None else True:
                try:self.statement(self.field(n,'body'))
                except BreakLoop:break
                if n.type=='for_statement' and self.field(n,'update') is not None:self.eval(self.field(n,'update'))
        elif n.type=='break_statement':raise BreakLoop()
        else:self.stop('STATEMENT_UNSUPPORTED:'+n.type)
        if self.witness is not None:self.stop('ADDRESS_POTENTIAL_WITNESS_FOUND')

    def analyze(self):
        result={'schema':'source_address_domain_v1','source_sha256':hashlib.sha256(self.src).hexdigest(),
            'unit_id':self.uid,'query_span':{'start_byte':self.target_span[0],'end_byte':self.target_span[1]},
            'potential_verdict':'UNKNOWN','observed_address_witness':None,'reachability':'NOT_PROVED',
            'full_C_program_safety':'UNKNOWN','other_C_UB_preconditions':'NOT_DISCHARGED',
            'domain':'finite source control/address trace with all memory contents UNKNOWN',
            'not_native_Eva_facts':True,'not_program_negative':True,'source_execution':False}
        try:
            if not self.abi_ok:self.stop('ABI_NOT_VERIFIED')
            if not source_tree_acceptable(self.tree,self.src):self.stop('SOURCE_PARSE_UNSUPPORTED')
            for n in nodes(self.tree.root_node):
                if n.type.startswith('preproc_') and n.type!='preproc_include':self.stop('SOURCE_DIRECTIVE_UNSUPPORTED')
                if n.type=='preproc_include' and not self.lib_ok:self.stop('INCLUDE_BINDING_NOT_VERIFIED')
                if n.type=='comment' and self.text(n).startswith(('/*@','//@')):self.stop('SOURCE_ACSL_UNSUPPORTED')
            definitions=[]
            for n in self.tree.root_node.named_children:
                if n.type!='function_definition':continue
                name=self.field(n,'declarator')
                while name is not None and name.type!='identifier':name=self.field(name,'declarator')
                if name is None:self.stop('SOURCE_FUNCTION_NAME_UNRESOLVED')
                if name and self.text(name) in ('malloc','free','assert'):self.stop('SOURCE_DEFINED_LIBRARY_NAME')
                if name and self.text(name)==self.entry:definitions.append(n)
            if len(definitions)!=1:self.stop('ENTRY_NOT_UNIQUE')
            body=self.field(definitions[0],'body');targets=[n for n in nodes(body) if (n.start_byte,n.end_byte)==self.target_span and
                (n.type=='subscript_expression' or (n.type=='pointer_expression' and self.text(self.field(n,'operator'))=='*'))]
            if len(targets)!=1:self.stop('QUERY_NOT_SINGLE_EXACT_SOURCE_ACCESS')
            target=targets[0]
            if any(n.type in ('assignment_expression','update_expression','call_expression') for n in nodes(target)):
                self.stop('QUERY_ADDRESS_SIDE_EFFECTS_UNSUPPORTED')
            if self.unevaluated(target) or self.formation(target):self.stop('QUERY_NOT_EVALUATED_ACCESS')
            stmt=target
            while stmt.parent and stmt.type not in ('expression_statement','declaration','return_statement'):stmt=stmt.parent
            accesses=[n for n in nodes(stmt) if n.type=='subscript_expression' or (n.type=='pointer_expression' and self.text(self.field(n,'operator'))=='*')]
            if len(accesses)!=1:self.stop('QUERY_STATEMENT_MULTIPLE_MEMORY_TARGETS')
            self.statement(body)
        except ReturnEntry:self.unknown.append('ENTRY_RETURNED')
        except BreakLoop:self.unknown.append('BREAK_OUTSIDE_SUPPORTED_LOOP')
        except Stop as exc:
            self.unknown.append(str(exc))
            if str(exc)!='ADDRESS_POTENTIAL_WITNESS_FOUND':self.witness=None
        if self.witness is not None:result['potential_verdict']=1
        result.update({'observed_address_witness':self.witness,'steps':self.steps,'budget':self.budget,
            'trace':self.trace,'unknown_reasons':self.unknown,'memory_content_reads':self.reads,'assumptions':self.assumptions,
            'ABI_sha256':self.abi['sha256'],'library_binding_sha256':self.binding.get('sha256') if self.lib_ok else None,
            'implementation_sha256':sha(__file__),'typed_kernel_sha256':sha(Path(__file__).with_name('evidence_numeric.py'))})
        return result


def analyze_address(source,unit_id,query_span,entry='main',library_binding=None,budget=512):
    if not isinstance(budget,int) or not 1<=budget<=10000:
        return {'potential_verdict':'UNKNOWN','failure':'INVALID_BUDGET','full_C_program_safety':'UNKNOWN'}
    try:return AddressDomain(source,unit_id,query_span,entry,library_binding,budget).analyze()
    except Exception as exc:
        return {'schema':'source_address_domain_v1','source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'unit_id':unit_id,'query_span':query_span,'potential_verdict':'UNKNOWN',
            'failure':type(exc).__name__+':'+str(exc),'full_C_program_safety':'UNKNOWN','source_execution':False}

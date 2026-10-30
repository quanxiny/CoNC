"""Finite author-token candidate graph. Not a C parser, solver or interpreter.

Only literal/assignment/add/sub/comparison candidates; unsupported operations
are retained as opaque nodes. Exact values below are rule/diagnostic side data,
never a learned-model input except at literal leaves.
"""
import re
OPS={'CONST':1,'COPY':2,'ADD':3,'SUB':4,'LT':5,'LE':6,'GT':7,'GE':8,'EQ':9,'NE':10,'MUL_UNKNOWN':11,'OPAQUE':12,'ARRAY':14,'MALLOC':15,'ACCESS':16,'COPY_DEST':17,'COPY_SOURCE':18,'PHI':19}
COMPS={'<':'LT','<=':'LE','>':'GT','>=':'GE','==':'EQ','!=':'NE'}
INTEGER=re.compile(r'^[+-]?\d+$');VAR=re.compile(r'^(VAR|FUN)\d+$')
class Builder:
    def __init__(self,lines,key,max_nodes=64):
        self.lines=lines;self.key=key;self.nodes=[];self.versions={};self.objects={};self.candidates=[];self.stats={'opaque_expressions':0,'branch_assignments':0,'mutations_unknown':0,'node_cap':0};self.max=max_nodes;self.line=0;self.cond=0
    def node(self,op,parents=(),literal=None,unit=0):
        if len(self.nodes)>=self.max:self.stats['node_cap']+=1;return 0
        idx=len(self.nodes);exact=literal
        if op=='COPY' and parents:exact=self.nodes[parents[0]]['exact']
        if op in ['ADD','SUB'] and len(parents)==2:
            a,b=[self.nodes[p]['exact'] for p in parents]
            if a is not None and b is not None:exact=a+b if op=='ADD' else a-b
        if op in ['PHI','OPAQUE','MUL_UNKNOWN']:exact=None
        n={'op':OPS[op],'op_name':op,'parents':list(parents[:2]),'literal':literal if op=='CONST' else None,
           'initial_known':op=='CONST' and literal is not None,'exact':exact,'line':self.line,'condition':self.cond,
           'unit':unit,'target':int(self.line==self.key),'binding':'conditional_text_candidate_not_C_proof'}
        self.nodes.append(n);return idx
    @staticmethod
    def trim(t):
        t=list(t)
        while t and t[-1] in [';',',']:t.pop()
        while len(t)>=2 and t[0]=='(' and t[-1]==')':
            depth=0;wrapped=True
            for i,x in enumerate(t):
                depth+=x=='(';depth-=x==')'
                if depth==0 and i<len(t)-1:wrapped=False;break
            if not wrapped:break
            t=t[1:-1]
        return t
    def expr(self,t):
        t=self.trim(t)
        if len(t)==1 and INTEGER.fullmatch(t[0]):return self.node('CONST',literal=int(t[0]))
        if len(t)==1 and VAR.fullmatch(t[0]):
            return self.node('COPY',[self.versions[t[0]]]) if t[0] in self.versions else self.opaque()
        if t==['sizeof','(','char',')']:return self.node('CONST',literal=1,unit=1)
        depth=0
        for i in range(len(t)-1,-1,-1):
            x=t[i];depth+=x==')';depth-=x=='('
            if depth==0 and x in ['+','-'] and i>0:return self.node('ADD' if x=='+' else 'SUB',[self.expr(t[:i]),self.expr(t[i+1:])])
        # sizeof(char)==1 is a declared unit normalization, shared by all arms.
        if len(t)>=5 and t[-5:]==['*','sizeof','(','char',')']:return self.node('COPY',[self.expr(t[:-5])],unit=1)
        if '*' in t:
            at=t.index('*');return self.node('MUL_UNKNOWN',[self.expr(t[:at]),self.expr(t[at+1:])])
        return self.opaque()
    def opaque(self):self.stats['opaque_expressions']+=1;return self.node('OPAQUE')
    @staticmethod
    def arguments(t,start):
        if start>=len(t) or t[start]!='(':return None
        depth=0;parts=[];part=[]
        for x in t[start+1:]:
            if x==')' and depth==0:parts.append(part);return parts
            if x==',' and depth==0:parts.append(part);part=[];continue
            depth+=x in ['(','['];depth-=x in [')',']'];part.append(x)
        return None
    def build(self):
        self.node('OPAQUE') # index0 is an explicit unknown sentinel, not zero.
        for li,raw in enumerate(self.lines):
            self.line=li;self.cond=2 if '<T>' in raw else 3 if '<F>' in raw else 1
            t=[x for x in raw if x not in ['<T>','<F>','<U>','\t']]
            if not t:continue
            is_decl=any(x in ['char','int','short','long','float','double','unsigned','signed'] for x in t[:3])
            if '=' in t:
                eq=t.index('=');lhs=[t[eq-1]] if eq and VAR.fullmatch(t[eq-1]) and '[' not in t[:eq] else []
                if lhs:
                    name=lhs[-1];rhs=t[eq+1:];parent=self.expr(rhs)
                    if 'malloc' in rhs:
                        mi=rhs.index('malloc');args=self.arguments(rhs,mi+1)
                        cap=self.expr(args[0]) if args and len(args)==1 else self.opaque()
                        obj=self.node('MALLOC',[cap],unit=1 if 'char' in t else 0);self.objects[name]=(cap,obj,1 if 'char' in t else 0)
                    new=self.node('COPY',[parent])
                    if self.cond in [2,3] or 'for' in t or 'while' in t:
                        self.stats['branch_assignments']+=1;new=self.node('PHI',[new,self.versions.get(name,0)])
                    self.versions[name]=new
            # Mutations and loop-carried assignments cannot retain a literal binding.
            if any(x in t for x in ['+=','-=','++','--','*=','/=']):
                for at,x in enumerate(t):
                    if VAR.fullmatch(x) and ((at+1<len(t) and t[at+1] in ['+=','-=','++','--','*=','/=']) or (at and t[at-1] in ['++','--'])):
                        self.versions[x]=self.opaque();self.stats['mutations_unknown']+=1
            if is_decl and '[' in t:
                a=t.index('[')
                if a and VAR.fullmatch(t[a-1]) and ']' in t[a+1:]:
                    b=t.index(']',a+1);cap=self.expr(t[a+1:b]);obj=self.node('ARRAY',[cap],unit=1 if 'char' in t else 2);self.objects[t[a-1]]=(cap,obj,1 if 'char' in t else 2)
            if not is_decl:
                for a in [i for i,x in enumerate(t) if x=='[' and i and VAR.fullmatch(t[i-1])]:
                    if ']' not in t[a+1:]:continue
                    b=t.index(']',a+1);index=self.expr(t[a+1:b]);cap,obj,unit=self.objects.get(t[a-1],(0,0,0))
                    use=self.node('ACCESS',[index,cap],unit=unit)
                    if li==self.key:self.candidates.append({'node':use,'kind':'index','a':index,'b':cap,'unit_known':unit>0})
            for api in ['memcpy','memmove','strncpy','strcpy','memset']:
                if api not in t:continue
                at=t.index(api);args=self.arguments(t,at+1)
                if not args or len(args)<2:continue
                count=self.expr(args[2]) if len(args)>=3 else self.opaque()
                roles=[(0,'COPY_DEST')]+([(1,'COPY_SOURCE')] if api in ['memcpy','memmove','strncpy','strcpy'] else [])
                for pos,role in roles:
                    name=args[pos][0] if len(args[pos])==1 else None;cap,obj,unit=self.objects.get(name,(0,0,0))
                    use=self.node(role,[count,cap],unit=unit)
                    if li==self.key:self.candidates.append({'node':use,'kind':'copy','a':count,'b':cap,'unit_known':unit==1})
            # Numeric comparisons are candidate operator nodes; no branch proof.
            for at,x in enumerate(t):
                if x in COMPS:
                    left=t[:at];right=t[at+1:]
                    if 'if' in left:left=left[left.index('if')+1:]
                    if left and left[0]=='(':left=left[1:]
                    if right and right[-1]==')':right=right[:-1]
                    self.node(COMPS[x],[self.expr(left),self.expr(right)])
        states=[];evidence=[]
        for c in self.candidates:
            a,b=self.nodes[c['a']]['exact'],self.nodes[c['b']]['exact']
            state='UNKNOWN' if self.stats['node_cap'] or a is None or b is None or not c['unit_known'] else ('VIOLATED' if a<0 or b<0 or (a>=b if c['kind']=='index' else a>b) else 'SATISFIED')
            states.append(state);evidence.append({**c,'conditional_operand_values':[a,b],'relation':state})
        decision='alert' if 'VIOLATED' in states else ('no_violation_proved_under_conditions' if states and all(s=='SATISFIED' for s in states) and not self.stats['node_cap'] else 'unknown')
        return {'nodes':self.nodes,'rules':{'state':decision,'evidence':evidence,'function_safety_truth':'UNKNOWN'},'stats':self.stats,'limits':{'nodes':self.max},'version_scope':'source-order candidates; branches retain PHI ambiguity; loop/alias/type proof unsupported'}
def build(lines,key):return Builder(lines,key).build()

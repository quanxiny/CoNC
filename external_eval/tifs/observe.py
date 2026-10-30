"""Conditional arithmetic probes on the exact supplied graph, during the one scoring pass."""
def reference(x):
    n=int(x['node_count']);values=[None]*n;depth=[0]*n;carry=[None]*n
    for i in range(n):
        op=int(x['ops'][i]);a,b=map(int,x['parents'][i])
        if op==1 and x['known'][i]:values[i]=sum(int(v)<<j for j,v in enumerate(x['bits'][i]))
        elif op==2 and 0<a<i:values[i]=values[a];depth[i]=depth[a]+1
        elif op in (3,4) and 0<a<i and 0<b<i and values[a] is not None and values[b] is not None:
            v=values[a]+values[b] if op==3 else values[a]-values[b]
            if -(2**31)<=v<2**31:values[i]=v;depth[i]=max(depth[a],depth[b])+1
            if values[b]==1 and values[a]>=0:
                z=values[a];run=0;bit=1 if op==3 else 0
                while run<32 and (z&1)==bit:run+=1;z>>=1
                carry[i]=run
    return values,depth,carry

def summarize(x,q,layer):
    import numpy as np
    values,depth,carry=reference(x);rows=[]
    for i,value in enumerate(values):
        if value is None:continue
        op=int(x['ops'][i]);literal=bool(x['known'][i]);truth=np.array([((value&0xffffffff)>>b)&1 for b in range(32)])
        soft=q[i,:32];pred=soft>=.5
        a,b=map(int,x['parents'][i]);operands=[values[j] for j in [a,b] if 0<=j<len(values) and values[j] is not None]
        width=32 if value<0 or any(v<0 for v in operands) else max([1,value.bit_length()]+[v.bit_length() for v in operands])
        rows.append({'layer':layer,'node':i,'operation':op,'literal':literal,'dependency_depth':depth[i],'carry_or_borrow_length':carry[i],
                     'reference_value':value,'integer_exact':bool(np.array_equal(pred,truth)),'all_bit_accuracy':float((pred==truth).mean()),
                     'effective_bit_accuracy':float((pred[:width]==truth[:width]).mean()),'effective_width':width,
                     'literal_max_abs_drift':float(abs(soft-truth).max()) if literal else None,
                     'interpretation':'conditional_on_supplied_graph; not verified C execution'})
    return rows

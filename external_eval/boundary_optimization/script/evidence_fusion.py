"""P3 scope-qualified evidence fusion; never turn a local bound into global safety."""
def qualified_x(state):
    if not state:return None
    cert=state.get('coverage_certificate',{})
    if state.get('decision')==1:
        # Positive must still come from verified source target binding / declared support.
        return 1 if state.get('positive_applicable',False) else None
    complete=all(cert.get(k) is True for k in ['all_source_targets_enumerated','all_targets_bound','related_preconditions_verified'])
    return 0 if state.get('decision')==0 and complete else None

def combine_scoped(x,e,*,compatible=True):
    x_value=qualified_x(x);e_value=e.get('program_target_decision')
    if not compatible:return {'decision':None,'reason':'INCOMPATIBLE_SOURCE_ENTRY_ABI','full_C_program_safety':'UNKNOWN'}
    if x_value==1 or e_value==1:
        value=1;reason='QUALIFIED_POTENTIAL_TARGET_ALARM'
    elif e.get('context_blocker_count',0)>0 or e.get('unresolved_target_count',0)>0:
        value=None;reason='UNRESOLVED_RELATED_CONTEXT_OR_TARGET'
    elif x_value==0:
        value=0;reason='INDEPENDENT_COMPLETE_X_SCOPE_NEGATIVE_E_FAILURE_NOT_USED_AS_NEGATIVE' if e.get('reported_evidence_status')=='NATIVE_FAILURE_OR_INCOMPLETE' else 'COMPLETE_X_SCOPE_NEGATIVE'
    elif e_value==0:
        value=0;reason='COMPLETE_E_SCOPE_NEGATIVE'
    else:value=None;reason='INCOMPLETE_SCOPE_OR_PRECONDITIONS'
    return {'decision':value,'reason':reason,'qualified_X':x_value,'qualified_E':e_value,'full_C_program_safety':'UNKNOWN'}

def fixtures():
    complete={'decision':0,'coverage_certificate':{k:True for k in ['all_source_targets_enumerated','all_targets_bound','related_preconditions_verified']}}
    local={'decision':0,'coverage_certificate':{}}
    absent={'program_target_decision':None,'context_blocker_count':0,'unresolved_target_count':0,'reported_evidence_status':'NATIVE_FAILURE_OR_INCOMPLETE'}
    init={**absent,'context_blocker_count':1,'reported_evidence_status':'CONTEXT_PRECONDITION_UNRESOLVED'}
    tests=[('local_X0_and_initialization_unknown',local,init,True,None),('complete_X0_and_initialization_unknown',complete,init,True,None),
        ('independent_complete_X0_E_native_failure_no_additional_evidence',complete,absent,True,0),('local_X0_E_native_failure',local,absent,True,None),
        ('X0_E_potential',complete,{**absent,'program_target_decision':1},True,1),
        ('X1_E0',{'decision':1,'positive_applicable':True},{**absent,'program_target_decision':0},True,1),
        ('unbound_X1',{'decision':1,'positive_applicable':False},absent,True,None),
        ('complete_X0_incompatible',complete,absent,False,None),('U_U',None,absent,True,None),
        ('X0_E_dead_target',complete,{**absent,'unresolved_target_count':1},True,None)]
    out=[]
    for name,x,e,compatible,want in tests:
        r=combine_scoped(x,e,compatible=compatible);assert r['decision']==want,(name,r)
        out.append({'fixture':name,'X':x,'E':e,'compatible':compatible,'expected':want,'actual':r})
    return out
if __name__=='__main__':
    import json
    print(json.dumps({'status':'PASS','fixtures':len(fixtures())}))

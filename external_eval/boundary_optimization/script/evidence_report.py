"""P0: target obligations, model assumptions and program proof are distinct."""
from pathlib import Path
import sys,re
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from external_eval.boundary_optimization.common import sha,digest
DEST=ROOT/'external_eval/tifs_runs/boundary_evidence_20261021'
POTENTIAL={'Unknown','Invalid','Invalid or unreachable'}
CONTRACT_KINDS={'assigns clause','from clause','postcondition'}

def obligation(row,config):
    kind=row.get('property kind','');p=row.get('property','');prefix=p.split(':',1)[0].strip()
    if kind=='index_bound':return 'target_array_bounds'
    if kind=='mem_access':return 'target_memory_validity'
    if kind.startswith('precondition of '):
        fn=kind.removeprefix('precondition of ')
        if prefix=='valid_stream':return 'context_stream'
        if fn in config['memory_functions'] and prefix in config['memory_precondition_names']:return 'target_library_range'
        return 'context_precondition'
    if kind in ['initialization','signed_overflow','unsigned_overflow','division_by_zero','pointer_alignment','pointer_value','finite_float']:
        return 'context_runtime'
    return 'other_obligation'

def contract_assumption(row,registry):
    """Pinned standard specification assumption, NOT exported dependency proof."""
    if row.get('status')!='Considered valid' or row.get('property kind') not in CONTRACT_KINDS:return None
    for entry in registry['functions']:
        if row.get('directory')!='FRAMAC_SHARE/libc' or row.get('file')!=entry['file'] or row.get('function')!=entry['function']:continue
        try:line=int(row['line'])
        except (ValueError,KeyError):return None
        if not entry['contract_start_line']<=line<=entry['declaration_end_line']:continue
        if sha(ROOT/entry['specification_path'])!=entry['specification_sha256']:raise ValueError('trusted specification changed')
        return {'function':entry['function'],'signature':entry['signature'],'specification_sha256':entry['specification_sha256'],
            'contract_start_line':entry['contract_start_line'],'declaration_end_line':entry['declaration_end_line'],
            'assumption':'trust pinned standard library model; actual call dependency not exported','dependency_chain_verified':False}
    return None

def classify_report(rows,config,registry,*,completed=True,failure=None,context=None,coverage_certificate=None):
    context=context or {};classified=[];targets=[];blockers=[];assumptions=[]
    context_compatible=context.get('source_and_entry_verified',False) and context.get('abi_verified',False)
    for row in rows:
        category=obligation(row,config);assumption=contract_assumption(row,registry);item={'row':row,'category':category,'assumption':assumption}
        classified.append(item)
        if category.startswith('target_'):targets.append(item)
        elif row.get('status')!='Valid' and assumption is None:blockers.append(item)
        if assumption:assumptions.append(assumption)
    potential=[a for a in targets if a['row'].get('status') in POTENTIAL]
    unresolved=[a for a in targets if a['row'].get('status')!='Valid']
    if failure or not completed:reported=None;status='NATIVE_FAILURE_OR_INCOMPLETE'
    elif not context_compatible:reported=None;status='CONTEXT_IDENTITY_UNVERIFIED'
    elif potential:reported=1;status='POTENTIAL_TARGET_ALARM'
    elif not targets:reported=None;status='NO_REPORTED_TARGETS'
    elif unresolved:reported=None;status='TARGET_DEAD_OR_UNRESOLVED'
    elif blockers:reported=None;status='CONTEXT_PRECONDITION_UNRESOLVED'
    else:reported=0;status='REPORTED_TARGETS_VALID_UNDER_LISTED_MODEL_ASSUMPTIONS' if assumptions else 'REPORTED_TARGETS_VALID'
    # TSV does not certify source access coverage, object binding or call dependency closure.
    certified=bool(coverage_certificate and coverage_certificate.get('all_source_targets_enumerated') is True and coverage_certificate.get('all_targets_bound') is True
       and coverage_certificate.get('related_preconditions_verified') is True and coverage_certificate.get('source_sha256')==context.get('source_sha256')
       and coverage_certificate.get('entry')==context.get('entry') and coverage_certificate.get('abi_hash')==context.get('abi_hash'))
    program=1 if reported==1 else 0 if reported==0 and certified and not assumptions else None
    return {'reported_target_decision':reported,'program_target_decision':program,'reported_evidence_status':status,
        'reported_target_count':len(targets),'potential_target_count':len(potential),'unresolved_target_count':len(unresolved),
        'context_blocker_count':len(blockers),'assumptions':assumptions,'obligations':classified,'context_blockers':blockers,
        'source_target_coverage_certified':certified,'negative_scope':'reported target properties only; not every source memory access',
        'entry_ABI_identity_verified':context_compatible,'call_dependency_chain_exported':False,'failure_reason':failure,
        'full_C_program_safety':'UNKNOWN','analysis_context':context}

def combine_evidence(x,e,*,compatible=True):
    """Only qualified scope evidence; keep U, never use probability to resolve evidence."""
    if not compatible:return None,'INCOMPATIBLE_CONTEXT'
    if x==1 or e==1:return 1,'TARGET_ALARM'
    if x==0 or e==0:return 0,'QUALIFIED_SCOPE_NEGATIVE'
    return None,'UNRESOLVED'

"""Process-local adapter over the unchanged, frozen family continuation loop."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[3];S=Path(__file__).parent;sys.path[:0]=[str(ROOT),str(S)]
from external_eval.boundary_optimization.common import read,sha,atomic_json
from classmatch_control import ARM,constants,classmatch_ce
import hybrid_retrain_train as trainer
P=ROOT/'external_eval/tifs_runs/hybrid_retraining_20261021'
CAMPAIGN=ROOT/'external_eval/tifs_runs/hybrid_evidence_20261021'
D=CAMPAIGN/'classmatch'
ORIGINAL_AUDIT=trainer.gate_label_audit


def verify_parent():
    protocol=read(CAMPAIGN/'classmatch_protocol.json')
    for n,h in protocol['files'].items():assert sha(ROOT/n)==h,n
    p=read(P/'freeze.json')
    for n,h in p['files'].items():assert sha(ROOT/n)==h,n
    for n,h in read(CAMPAIGN/'classmatch_parent_protection.json')['files'].items():assert sha(ROOT/n)==h,n
    for a in read(P/'selection.json')['results']:
        folder=Path(a['checkpoint']).parent
        assert sha(folder/'completion.json')==a['completion_sha256']
        c=read(folder/'completion.json');assert c['steps']==1280
        for name in ['best','last']:assert sha(folder/(name+'.pt'))==c[name+'_sha256']
        assert c['best_sha256']==a['checkpoint_sha256']
    return p


def verify_freeze():
    parent=verify_parent();f=read(D/'freeze.json');cfg=read(D/'config.json')
    for n,h in f['files'].items():assert sha(ROOT/n)==h,n
    assert f['config_sha256']==sha(D/'config.json') and f['parent_freeze_sha256']==sha(P/'freeze.json')
    assert f['base_models']==parent['base_models'] and f['parameter_contract']==parent['parameter_contract']
    previous=read(P/'config.json')
    for key in ['seeds','learning_rate','weight_decay','clip_grad_norm','batch_size','updates','steps_per_validation_cycle','validation_cycles','precision','p_R','mu']:
        assert cfg[key]==previous[key],key
    c=read(CAMPAIGN/'classmatch_protocol.json')['constants_TRAIN_only']
    assert cfg['classmatch_constants']==c and cfg['mu']==c['mu'] and cfg['arms']==[ARM]
    for name in ['gates','schedules']:
        assert (D/name).is_symlink() and (D/name).resolve()==(P/name).resolve()
    return f,cfg


def install(cfg):
    """Only module globals in this isolated worker change; core/disk stay frozen."""
    c=cfg['classmatch_constants'];assert c==constants(c['class_counts'],c['residual_counts'])
    def loss(logits,labels,R,arm,mu):
        assert mu==cfg['mu']==c['mu']
        return classmatch_ce(logits,labels,R,arm,mu,c['class_weights'])
    def audit(gates):
        a=ORIGINAL_AUDIT(gates)
        for split in ['train','val']:
            r=a[split];r['residual_reference_weighted_positive_fraction']=r['weighted_positive_fraction']
            neg=r['R0_label0']+r['R1_label0'];pos=r['R0_label1']+r['R1_label1']
            r['weighted_positive_fraction']=pos*c['class_weights'][1]/(neg*c['class_weights'][0]+pos*c['class_weights'][1])
            r['weight_definition']='class constants fitted from full TRAIN only; DEV labels used for descriptive audit only'
            r['causal_limit']='class-prior-matched comparison; no within-class R weighting; not proof of learned numerical operations'
        a['classmatch_constants']=c
        return a
    trainer.D=D;trainer.ARMS=(ARM,);trainer.supervised_ce=loss
    trainer.gate_label_audit=audit;trainer.verify_freeze=verify_freeze
    return trainer


def run(seed,device):
    f,cfg=verify_freeze();loop=install(cfg)
    assert (D/'training').resolve().is_relative_to(CAMPAIGN.resolve())
    return loop.run(ARM,seed,device)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[11,29,47],required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();run(a.seed,a.device)

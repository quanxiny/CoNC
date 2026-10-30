"""One frozen stage, checkpointed cursor/RNG, source DEV selection only."""
import argparse,fcntl,os,resource,signal,time
from dataclasses import asdict
from .common import *

def stage_dir(out,variant,seed,stage):return out/'training'/f'{variant}_s{seed}'/f'stage{stage}'

def run(out,variant,seed,stage,device='cuda:0',resume=False):
    out=output_dir(out);frozen=require_freeze(out);cfg=config()
    if variant not in cfg['variants']+['vd'] or seed not in cfg['seeds'] or stage not in [1,2]:raise ValueError('Unregistered job')
    if variant=='vd' and stage!=1:raise ValueError('VD is single-stage')
    _,torch=setup(seed,cfg['runtime']['cpu_threads'])
    from .models import make_model,transfer_stage1
    from overflow51.training import TrainConfig,Cursor,EarlyStop,snapshot,restore,atomic_checkpoint,load_own_checkpoint
    from torch.nn import functional as F
    data=SourceData()
    with (out/'source_manifest.csv').open() as stream:rows=list(csv.DictReader(stream))
    tr=[int(r['row']) for r in rows if r['split']=='train'];va=[int(r['row']) for r in rows if r['split']=='val']
    recipe=dict(cfg['training']['vd' if variant=='vd' else 'stage'+str(stage)])
    epoch0=recipe.pop('epoch0_eligible');tc=TrainConfig(seed=seed,**recipe)
    dest=stage_dir(out,variant,seed,stage);dest.mkdir(parents=True,exist_ok=True)
    lock=(dest/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity={'freeze_sha256':sha(out/'freeze.json'),'variant':variant,'seed':seed,'stage':stage,'config':asdict(tc),'epoch0_eligible':epoch0}
    model=make_model(variant,stage,seed)
    if stage==2:
        prior=stage_dir(out,variant,seed,1);completion=read(prior/'completion.json')
        if sha(prior/'best.pt')!=completion['best_sha256']:raise ValueError('Stage1 checkpoint changed')
        ck=load_own_checkpoint(prior/'best.pt');transfer_stage1(model,ck['model'])
        identity['stage1_best_sha256']=sha(prior/'best.pt');del ck
    atomic_json(dest/'identity.json',identity,immutable=True);mh=digest(identity)
    if (dest/'completion.json').exists():
        result=read(dest/'completion.json')
        if result['identity']!=identity or sha(dest/'best.pt')!=result['best_sha256']:raise ValueError('Completed run identity mismatch')
        return result
    if (dest/'last.pt').exists() and not resume:raise ValueError('Interrupted run requires explicit --resume')
    if not device.startswith('cuda'):raise ValueError('Formal source training requires registered CUDA runtime')
    torch.cuda.set_device(torch.device(device));torch.cuda.reset_peak_memory_stats()
    model=model.to(device)
    initial_frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
    opt=(torch.optim.Adam if variant=='vd' else torch.optim.AdamW)([p for p in model.parameters() if p.requires_grad],lr=tc.lr,weight_decay=tc.weight_decay)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=tc.scheduler_gamma)
    cursor=Cursor(len(tr),seed);stop=EarlyStop();steps=0;elapsed_before=0.;phase='initial';history=[]
    if (dest/'last.pt').exists():
        ck=load_own_checkpoint(dest/'last.pt');steps=restore(ck,model,opt,scheduler,cursor,stop,tc,mh)
        elapsed_before=ck['elapsed_s'];phase=ck['phase'];history=ck['history'];del ck
    start=time.monotonic();limit=cfg['runtime']['wall_seconds_per_stage']
    signal.alarm(max(1,int(limit-elapsed_before+cfg['runtime']['hard_grace_seconds'])))
    def elapsed():return elapsed_before+time.monotonic()-start
    def save(name):
        ck=snapshot(model,opt,scheduler,cursor,stop,tc,mh,steps)
        ck.update(elapsed_s=elapsed(),phase=phase,history=history)
        atomic_checkpoint(dest/name,ck)
    @torch.no_grad()
    def dev_loss():
        model.eval();loss=0.
        for at in range(0,len(va),tc.batch_size):
            x,y=data.batch(va[at:at+tc.batch_size],device,variant=='vd')
            logits=model(x)['logits']
            if not logits.isfinite().all():raise FloatingPointError('Invalid DEV logits')
            loss+=float(F.cross_entropy(logits,y,reduction='sum'))
        return loss/len(va)
    if phase=='initial':
        if epoch0:
            value=dev_loss();stop.best=value;stop.best_epoch=0
            history.append({'epoch':0,'dev_native_ce':value,'selected':True,'elapsed_s':elapsed()});save('best.pt')
        phase='ready';save('last.pt')
    exhausted=False
    while phase!='finished' and stop.bad_epochs<tc.patience:
        completed=cursor.epoch+int(cursor.position==cursor.size)
        if completed>=tc.max_epochs:break
        if elapsed()>=limit:exhausted=True;break
        model.train();phase='training'
        while True:
            if elapsed()>=limit:exhausted=True;break
            ids=cursor.next(tc.batch_size);x,y=data.batch([tr[int(i)] for i in ids],device,variant=='vd')
            opt.zero_grad(set_to_none=True);logits=model(x)['logits'];loss=F.cross_entropy(logits,y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),tc.grad_clip,error_if_nonfinite=True)
            opt.step();steps+=1
            if steps%50==0:
                progress={'phase':phase,'epoch':cursor.epoch+1,'position':cursor.position,'steps':steps,'loss':float(loss.detach()),'gradient_norm':float(norm),'elapsed_s':elapsed()}
                atomic_json(dest/'progress.json',progress);print(json.dumps(progress),flush=True)
            if steps%cfg['runtime']['checkpoint_every_steps']==0 and cursor.position<cursor.size:save('last.pt')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>cfg['runtime']['maximum_rss_gib']*1024**2:raise MemoryError('RSS budget exceeded')
            if torch.cuda.max_memory_allocated()>cfg['runtime']['maximum_gpu_gib']*2**30:raise MemoryError('GPU memory budget exceeded')
            if cursor.position==cursor.size:break
        if exhausted:save('last.pt');break
        scheduler.step();epoch=cursor.epoch+1;value=dev_loss();better,done=stop.update(value,epoch,tc)
        history.append({'epoch':epoch,'dev_native_ce':value,'selected':better,'steps':steps,'elapsed_s':elapsed()})
        phase='ready';save('last.pt')
        if better:save('best.pt')
        atomic_json(dest/'history.json',history);print(json.dumps(history[-1]),flush=True)
        if done:break
    if not (dest/'best.pt').exists():raise RuntimeError('No complete DEV-selected checkpoint; downstream blocked')
    for n,p in model.named_parameters():
        if n in initial_frozen and not torch.equal(p.detach().cpu(),initial_frozen[n]):raise ValueError('Frozen parameter changed: '+n)
    phase='finished';save('last.pt');signal.alarm(0)
    result={'status':'COMPLETE_BUDGET_LIMITED' if exhausted else 'COMPLETE','identity':identity,'steps':steps,'best_epoch':stop.best_epoch,
            'best_dev_native_ce':stop.best,'completed_epochs':cursor.epoch+int(cursor.position==cursor.size),'budget_exhausted':exhausted,
            'early_stopped':stop.bad_epochs>=tc.patience,'elapsed_s':elapsed(),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'parameters':sum(p.numel() for p in model.parameters()),
            'requires_grad_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'inactive_parameters_not_counted_as_capacity_match':True,
            'best_sha256':sha(dest/'best.pt'),'frozen_weights_unchanged':True,'test_used_for_selection':False,'external_used_for_selection':False}
    atomic_json(dest/'completion.json',result,immutable=True);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=campaign_root());p.add_argument('--variant',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--stage',type=int,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--resume',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.out,a.variant,a.seed,a.stage,a.device,a.resume)),flush=True)
if __name__=='__main__':main()

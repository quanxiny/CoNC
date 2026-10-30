"""Source-stage training on the archived family TRAIN/DEV allocation."""
from portable_runtime import configure_imports, ROOT as RELEASE_ROOT, CONTRACT, check_output, sha
configure_imports()
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('MKL_NUM_THREADS','8')
import argparse,json,time,subprocess,shutil,resource,signal,math
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from common70 import SelectedData
from all_visible52 import AllVisible
from overflow51.baselines import RiskBridge, LegacyPureGraph, NativeVD
from overflow51.overflowgen import original_vd_batch
ROOT = RELEASE_ROOT

class FullData(SelectedData):
    def __init__(self, track, arm):
        if track != 'family': raise ValueError('This release entry uses the historical family allocation')
        super().__init__(); self.arm = arm
        self.manifest = {'manifest_sha256': sha(CONTRACT)}
    def indices(self, split): return self.rows(split)
    def batch(self, rows, device='cpu'):
        x, y = super().batch(rows, device)
        if self.arm == 'vd':
            x = {k:v.to(device) for k,v in original_vd_batch(self.a, rows).items()}
        return x, y

def make_model(arm, track):
    if arm == 'numeric_all_visible': return RiskBridge(AllVisible())
    if arm == 'vd': return RiskBridge(NativeVD(track))
    return RiskBridge(LegacyPureGraph('joint' if arm == 'legacy_joint' else arm))
from overflow51.training import TrainConfig,EarlyStop,Cursor,seed_all,snapshot,restore,atomic_checkpoint,load_own_checkpoint,train_epoch,evaluate_batches

RUNROOT=Path(__file__).resolve().parent

ARMS=('numeric_all_visible', 'vd', 'semantic', 'numeric', 'legacy_joint')

def configuration(arm,seed,lr=None):
    return TrainConfig(seed=seed,lr=lr if lr is not None else 2e-5 if arm=='vd' else 1e-4,
        max_epochs=100,patience=10,min_delta=0. if arm=='vd' else 1e-4,batch_size=64,weight_decay=0. if arm=='vd' else .01,
        grad_clip=5.,scheduler_gamma=.95,positive_weight=1.,binding_weight=0.,path_weight=0.)


def main():
    global RUNROOT
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--track',choices=['family'],default='family')
    p.add_argument('--seed',type=int,choices=[11,29,47],required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--lr',type=float);p.add_argument('--hours',type=float,default=4);p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args();RUNROOT=check_output(args.output_dir);cfg=configuration(args.arm,args.seed,args.lr)
    if not 0<args.hours<=4:raise ValueError('each worker has a maximum four-hour checkpointed budget')
    spec={'arm':args.arm,'track':args.track,'config':asdict(cfg),'precision':'FP32',
      'n_training_rows':94179,'n_dev_rows':11940,'test_used_for_selection':False,'max_wall_hours':args.hours,
      'eval_status':'historical regression','device':'cuda:0'}
    print(json.dumps(spec,ensure_ascii=False),flush=True)
    if not args.execute:return
    from portable_runtime import validate_training_readiness
    validate_training_readiness()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for full source-stage training')
    signal.alarm(math.ceil(args.hours*3600+300))
    if shutil.disk_usage(ROOT).free < 8*2**30: raise RuntimeError('At least 8 GiB free disk space is required')
    device='cuda:0';torch.set_num_threads(4);seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=FullData(args.track,args.arm);train=data.indices('train');val=data.indices('val')
    model=make_model(args.arm,args.track).to(device)
    optimizer=(torch.optim.Adam if args.arm=='vd' else torch.optim.AdamW)(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=cfg.scheduler_gamma)
    cursor=Cursor(len(train),cfg.seed);stop=EarlyStop();steps=0;start=time.monotonic();deadline=start+args.hours*3600
    tag=f'{args.track}_{args.arm}_s{args.seed}_lr{cfg.lr}';out=RUNROOT/'runs'/tag
    if out.exists() and not args.resume:raise FileExistsError('run exists; use exact-config resume or new registered trial')
    out.mkdir(parents=True,exist_ok=True);mh=data.manifest['manifest_sha256']
    if args.resume:steps=restore(load_own_checkpoint(out/'last.pt'),model,optimizer,scheduler,cursor,stop,cfg,mh)
    (out/'config.json').write_text(json.dumps(spec|{'manifest_sha256':mh,'parameters':sum(p.numel() for p in model.parameters()),'host_pid':os.getpid(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'worker_sha256':__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    def get_batch(ix):return data.batch(train[ix.numpy()],device)
    def vbatches():
        for pos in range(0,len(val),cfg.batch_size):yield data.batch(val[pos:pos+cfg.batch_size],device)
    probe=train[:2048]
    def tbatches():
        for pos in range(0,len(probe),cfg.batch_size):yield data.batch(probe[pos:pos+cfg.batch_size],device)
    def save(name):atomic_checkpoint(out/name,snapshot(model,optimizer,scheduler,cursor,stop,cfg,mh,steps))
    with (out/'steps.jsonl').open('a') as slog,(out/'epochs.jsonl').open('a') as elog:
        def record(epoch,position,terms,counts,norm):
            nonlocal steps
            steps+=1
            if steps%20==0:slog.write(json.dumps({'step':steps,'epoch':epoch,'position':position,'loss':terms,'counts':counts,'grad_norm':norm,'elapsed_s':time.monotonic()-start})+'\n');slog.flush()
            if steps%250==0 and cursor.position<cursor.size:save('last.pt')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>48*1024**2:raise MemoryError('worker exceeded 48 GiB RSS budget')
        while cursor.epoch<cfg.max_epochs:
            if cursor.position==cursor.size and cursor.epoch+1>=cfg.max_epochs:break
            completed=train_epoch(model,optimizer,scheduler,cursor,cfg,get_batch,deadline=deadline,clock=time.monotonic,log_step=record)
            if not completed:save('last.pt');break
            vm=evaluate_batches(model,vbatches());tm=evaluate_batches(model,tbatches());v=vm['loss'];better,done=stop.update(v,cursor.epoch,cfg)
            elog.write(json.dumps({'epoch':cursor.epoch+1,'steps':steps,'val_loss':v,'best':better,'elapsed_s':time.monotonic()-start,
                'val_metrics':vm,'train_fixed2048_metrics':tm,'allocated_GiB':torch.cuda.max_memory_allocated(0)/2**30})+'\n');elog.flush()
            save('last.pt')
            if better:save('best.pt')
            if done:break
    (out/'completion.json').write_text(json.dumps({'steps':steps,'completed_epochs':cursor.epoch+int(cursor.position==cursor.size),
      'best_epoch':stop.best_epoch+1,'best_val_loss':stop.best,'early_stopped':stop.bad_epochs>=cfg.patience,
      'budget_exhausted':time.monotonic()>=deadline,'wall_s':time.monotonic()-start,
      'convergence_sufficiency':'inspect complete train/val curves; budget/epoch cap alone is not convergence','test_predictions':0,'host_pid':os.getpid()},indent=2))

    if args.arm == 'numeric_all_visible':
        checkpoint = load_own_checkpoint(out/'best.pt')
        model.load_state_dict(checkpoint['model'])
        metrics = evaluate_batches(model, vbatches())
        receipt = out/'DEV_metrics.json'
        receipt.write_text(json.dumps({'metrics':dict(metrics, NLL_original_arithmetic=metrics['loss'])},indent=2)+'\n')
        source = {'checkpoint':'best.pt', 'checkpoint_sha256':sha(out/'best.pt'),
                  'data_manifest_sha256':mh, 'DEV_receipt':'DEV_metrics.json'}
        (out/'transfer_sources.json').write_text(json.dumps({'transfer_sources':{str(args.seed):source}},indent=2)+'\n')

if __name__=='__main__':main()

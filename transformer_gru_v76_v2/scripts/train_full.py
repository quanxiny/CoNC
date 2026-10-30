from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from portable_runtime import configure_imports, ROOT as RELEASE_ROOT, CONTRACT, check_output
configure_imports()
"""CoNC and refiner controls on the historical family TRAIN/DEV split."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from common70 import *
from model70 import AdaptiveGraph70, ARMS
from model71 import FrozenResidual71
ARMS = dict(ARMS, residual_transfer_frozen=('residual', True, True))
def make_model(arm):
    return FrozenResidual71() if arm == 'residual_transfer_frozen' else AdaptiveGraph70(arm)
from portable_runtime import portable_protocol as protocol, validate_training_readiness
import argparse
import csv
import gzip
import math
import resource
import shutil
import signal
import time
from dataclasses import asdict
from torch.nn import functional as F
from overflow51.training import TrainConfig, EarlyStop, Cursor, seed_all, snapshot, restore, atomic_checkpoint, load_own_checkpoint
from overflow51.metrics import risk_metrics


def config(seed):
    return TrainConfig(seed=seed, max_epochs=100, patience=10, min_delta=1e-4,
        lr=1e-4, batch_size=64, weight_decay=.01, grad_clip=5., scheduler_gamma=.95)


def setup(seed):
    torch.set_num_threads(4); seed_all(seed)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def evaluate(model, data, rows, save=None, steps=6):
    was_training = model.training; model.eval(); ys = []; scores = []; loss_sum = 0.; diag = {}
    for start in range(0, len(rows), 64):
        ix = rows[start:start + 64]; x, y = data.batch(ix, 'cuda:0'); out = model(x, steps=steps)
        if not out['logits'].isfinite().all(): raise FloatingPointError('Nonfinite evaluation')
        loss_sum += float(F.cross_entropy(out['logits'], y, reduction='sum'))
        ys.extend(y.cpu().tolist()); scores.extend(out['risk'].cpu().tolist())
        for k, v in out['diagnostic'].items(): diag[k] = diag.get(k, 0.) + float(v) * len(ix)
    model.train(was_training)
    y = np.asarray(ys); p = np.asarray(scores); clipped = np.clip(p, 1e-7, 1-1e-7)
    result = {'native_NLL': loss_sum / len(rows), **risk_metrics(y, p),
        'score_NLL_clipped': float(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)).mean()),
        'diagnostics_row_weighted': {k: v / len(rows) for k, v in diag.items()},
        'residual_scale': float(model.residual_scale.detach().tanh()), 'computation_steps': steps}
    if save is not None:
        with gzip.open(save, 'xt') as f:
            w = csv.writer(f); w.writerow(['row', 'label', 'score'])
            w.writerows(zip(data.a['row'][rows].tolist(), ys, scores))
        result['prediction_sha256'] = sha(save)
    return result


def main():
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--arm', choices=['conc'] + list(ARMS), required=True,
        help='conc selects CoNC (historical liquid_transfer_frozen); other names identify historical controls')
    ap.add_argument('--seed', type=int, choices=[11,29,47], required=True)
    ap.add_argument('--execute', action='store_true'); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    if args.arm == 'conc': args.arm = 'liquid_transfer_frozen'
    ROOT = check_output(args.output_dir); spec = protocol(); cfg = config(args.seed)
    print(json.dumps({'paper_name': 'CoNC' if args.arm == 'liquid_transfer_frozen' else args.arm,
        'arm': args.arm, 'seed': args.seed, 'config': asdict(cfg), 'dry_run': not args.execute}), flush=True)
    if not args.execute: return
    validate_training_readiness()
    if shutil.disk_usage(RELEASE_ROOT).free < spec['resources']['minimum_free_bytes']: raise RuntimeError('Insufficient free disk space')
    ROOT.mkdir(parents=True,exist_ok=args.resume)
    (ROOT/'runs').mkdir(exist_ok=args.resume)
    limit = spec['resources']['training_wall_s_per_run']; signal.alarm(limit + 1800)
    resource.setrlimit(resource.RLIMIT_CPU, (limit*5, limit*5)); setup(args.seed)
    torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    data = SelectedData(); tr = data.rows('train'); va = data.rows('val')
    assert (len(tr), len(va)) == (94179,11940)
    model = make_model(args.arm)
    source = spec['transfer_sources'][str(args.seed)]
    mapping = None
    if ARMS[args.arm][1]:
        if sha(source['checkpoint']) != source['checkpoint_sha256']: raise ValueError('Source checkpoint changed')
        checkpoint = load_own_checkpoint(Path(source['checkpoint']))
        if checkpoint['manifest_sha256'] != source['data_manifest_sha256']: raise ValueError('Source preprocessing changed')
        mapping = model.transfer(checkpoint['model']); del checkpoint
    model = model.cuda()
    opt = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg.scheduler_gamma)
    cursor = Cursor(len(tr), args.seed); stop = EarlyStop(); steps = 0; before = 0.
    dest = ROOT / 'runs' / f'{args.arm}_s{args.seed}'
    if dest.exists() and not args.resume: raise FileExistsError('No duplicate experiment')
    if (dest/'completion.json').exists(): raise RuntimeError('Completed experiment cannot be resumed')
    dest.mkdir(exist_ok=args.resume)
    manifest = sha(CONTRACT) + ':' + args.arm
    if args.resume:
        ck = load_own_checkpoint(dest/'last.pt'); steps = restore(ck, model, opt, scheduler, cursor, stop, cfg, manifest)
        before = ck['elapsed_total_s']; del ck
        if before >= limit: raise RuntimeError('Cumulative budget already exhausted; no extension')
        if stop.bad_epochs>=cfg.patience or cursor.epoch+int(cursor.position==cursor.size)>=cfg.max_epochs:
            raise RuntimeError('Training stopping condition already reached; no additional updates')
    started = time.monotonic()
    def elapsed(): return before + time.monotonic() - started
    def save(name):
        ck = snapshot(model, opt, scheduler, cursor, stop, cfg, manifest, steps)
        ck['elapsed_total_s'] = elapsed(); atomic_checkpoint(dest/name, ck)
    frozen_initial = {n: v.detach().cpu().clone() for n,v in model.named_parameters() if not v.requires_grad}
    write(dest/('resume_config.json' if args.resume else 'config.json'), {'arm':args.arm,'seed':args.seed,
        'config':asdict(cfg),'parameters':sum(v.numel() for v in model.parameters()),
        'trainable_parameters':sum(v.numel() for v in model.parameters() if v.requires_grad),
        'precision':'FP32; TF32 disabled','source':source if mapping else None,'transfer_map':mapping,
        'optimizer_state_transferred':False,'shared_LR_all_trainable_parameters':True,
        'frozen_weight_names':list(frozen_initial),'frozen_module_dropout':'same training-mode dropout as other arms; weights only frozen',
        'manifest':manifest,'max_cumulative_wall_s':limit,'host_pid':os.getpid(),'test_rows':0})
    exhausted = False; finished = False
    with (dest/'epochs.jsonl').open('a') as el, (dest/'steps.jsonl').open('a') as sl:
        if not args.resume:
            vm = evaluate(model,data,va); stop.best=vm['native_NLL']; stop.best_epoch=0
            if mapping:
                ref=read(source['DEV_receipt'])['metrics']
                if abs(vm['native_NLL']-ref['NLL_original_arithmetic'])>1e-5 or abs(vm['f1_all_positive']-ref['f1_all_positive'])>1e-12:
                    raise ValueError('Complete DEV initial transfer reproduction failed before optimization')
            el.write(json.dumps({'epoch':0,'steps':0,'DEV':vm,'selected':True,'elapsed_s':elapsed(),'initial_checkpoint':True})+'\n');el.flush()
            save('best.pt');save('last.pt')
        while not finished and cursor.epoch < cfg.max_epochs:
            model.train(); complete=True; online_sum=0.; online_n=0
            while True:
                if elapsed() >= limit: complete=False; exhausted=True; break
                ix=cursor.next(cfg.batch_size); x,y=data.batch(tr[ix.numpy()],'cuda:0')
                opt.zero_grad(set_to_none=True);out=model(x);loss=F.cross_entropy(out['logits'],y)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                opt.step();steps+=1;online_sum+=float(loss.detach())*len(y);online_n+=len(y)
                if steps%50==0:
                    sl.write(json.dumps({'step':steps,'epoch':cursor.epoch+1,'loss':float(loss.detach()),
                        'global_gradient_before_clip':float(grad),'residual_scale':float(out['residual_scale'].detach()),
                        'diagnostic':{k:float(v.detach()) for k,v in out['diagnostic'].items()},'elapsed_s':elapsed()})+'\n');sl.flush()
                if steps%250==0: save('last.pt')
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>16*1024**2: raise MemoryError('16GiB RSS cap')
                if torch.cuda.max_memory_allocated()>32*2**30: raise MemoryError('32GiB allocated GPU cap')
                if cursor.position==cursor.size: break
            if not complete: save('last.pt');break
            scheduler.step();epoch=cursor.epoch+1;vm=evaluate(model,data,va)
            better,finished=stop.update(vm['native_NLL'],epoch,cfg)
            tm=evaluate(model,data,tr[:2048])
            el.write(json.dumps({'epoch':epoch,'steps':steps,'DEV':vm,'TRAIN_fixed2048':tm,
                'TRAIN_online_NLL':online_sum/online_n,'selected':better,'elapsed_s':elapsed()})+'\n');el.flush()
            save('last.pt')
            if better: save('best.pt')
            if epoch>=cfg.max_epochs: break
    for n,v in model.named_parameters():
        if n in frozen_initial and not torch.equal(v.detach().cpu(),frozen_initial[n]): raise ValueError('Frozen parameter changed: '+n)
    train_wall=elapsed(); checkpoint=load_own_checkpoint(dest/'best.pt');model.load_state_dict(checkpoint['model']);del checkpoint
    # Separate bounded inference allowance; no further optimizer updates.
    signal.alarm(1800)
    final={'DEV':evaluate(model,data,va,dest/'DEV_predictions.csv.gz'),
           'TRAIN':evaluate(model,data,tr,dest/'TRAIN_predictions.csv.gz')}
    for k in [3,12]: final[f'DEV_depth{k}']=evaluate(model,data,va,steps=k)
    result={'status':'COMPLETE','arm':args.arm,'seed':args.seed,'steps':steps,
        'completed_epochs':cursor.epoch+int(cursor.position==cursor.size),'best_epoch':stop.best_epoch,
        'early_stopped':stop.bad_epochs>=cfg.patience,'budget_exhausted':exhausted or train_wall>=limit,
        'max_epochs_reached':cursor.epoch+int(cursor.position==cursor.size)>=cfg.max_epochs,
        'training_wall_s':train_wall,'total_wall_s_including_final_inference':elapsed(),
        'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'peak_GPU_GiB':torch.cuda.max_memory_allocated()/2**30,'test_rows':0,
        'best_checkpoint_sha256':sha(dest/'best.pt'),'frozen_weights_unchanged':True,'metrics':final}
    write(dest/'completion.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__': main()

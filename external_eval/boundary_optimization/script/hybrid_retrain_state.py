"""RNG/checkpoint helpers shared by the real trainer and CPU resume fixture."""
from pathlib import Path
import argparse, hashlib, json, os, random


def capture_rng(np, torch, device=None):
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': None if device is None or str(device).startswith('cpu') else torch.cuda.get_rng_state(device)}


def restore_rng(saved, np, torch, device=None):
    random.setstate(saved['python']); np.random.set_state(saved['numpy']); torch.set_rng_state(saved['torch'])
    if saved['cuda'] is not None:
        if device is None or str(device).startswith('cpu'):
            raise ValueError('CUDA RNG checkpoint requires original device type')
        torch.cuda.set_rng_state(saved['cuda'], device)


def atomic_torch_save(path, state, torch):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    torch.save(state, tmp)
    with tmp.open('rb') as f: os.fsync(f.fileno())
    os.replace(tmp, path)


def repair_step_log(path, committed_step):
    """Preserve an uncommitted tail before deterministic replay, never discard it."""
    p = Path(path)
    if not p.exists():
        if committed_step: raise ValueError('checkpoint is ahead of step ledger')
        return
    raw = p.read_bytes(); lines = raw.splitlines(keepends=True)
    if len(lines) < committed_step:
        raise ValueError('incomplete committed step ledger')
    for j, line in enumerate(lines[:committed_step], 1):
        if not line.endswith(b'\n'):
            raise ValueError('unterminated committed step record')
        row = json.loads(line)
        if type(row.get('step')) is not int or row['step'] != j:
            raise ValueError('noncontiguous committed step ledger')
    prefix = b''.join(lines[:committed_step]); tail = raw[len(prefix):]
    if tail:
        tag = hashlib.sha256(tail).hexdigest()[:16]
        archive = p.with_name(p.stem + '_uncommitted_' + tag + p.suffix)
        if archive.exists():
            if archive.read_bytes() != tail: raise ValueError('recovery tail mismatch')
        else:
            temporary = archive.with_suffix(archive.suffix + '.tmp')
            with temporary.open('wb') as stream:
                stream.write(tail); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, archive)
        tmp = p.with_suffix('.recovery.tmp')
        with tmp.open('wb') as stream:
            stream.write(prefix); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, p)


def toy_resume_check(folder):
    import numpy as np
    import torch
    from hybrid_retrain_core import supervised_ce
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    def start():
        random.seed(711); np.random.seed(711); torch.manual_seed(711)
        model = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.Dropout(.3), torch.nn.Tanh(), torch.nn.Linear(6, 2))
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
        schedule = np.random.default_rng(811).integers(0, 12, (4, 4)).tolist()
        return model, opt, schedule
    X = torch.arange(36, dtype=torch.float32).reshape(12, 3) / 10
    Y = torch.arange(12).remainder(2).long(); R = torch.arange(12).remainder(2).bool()
    def steps(model, opt, schedule, lo, hi, ledger):
        for j in range(lo, hi):
            ids = schedule[j]; model.train(); opt.zero_grad(set_to_none=True)
            logits = model(X[ids]); loss, _ = supervised_ce(logits, Y[ids], R[ids], 'residual_weighted', 1.5)
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            opt.step(); ledger.append({'step': j + 1, 'batch': ids, 'loss': float(loss),
                'grad_norm': float(norm), 'logits': logits.detach().tolist()})
    a, oa, schedule = start(); whole = []; steps(a, oa, schedule, 0, 4, whole)
    expected = {'model': a.state_dict(), 'optimizer': oa.state_dict(), 'rng': capture_rng(np, torch), 'ledger': whole}
    b, ob, schedule = start(); partial = []; steps(b, ob, schedule, 0, 2, partial)
    path = folder / 'toy_last.pt'
    atomic_torch_save(path, {'model': b.state_dict(), 'optimizer': ob.state_dict(), 'rng': capture_rng(np, torch),
        'step': 2, 'schedule': schedule, 'ledger': partial}, torch)
    c, oc, _ = start(); saved = torch.load(path, map_location='cpu', weights_only=False)
    c.load_state_dict(saved['model']); oc.load_state_dict(saved['optimizer']); restore_rng(saved['rng'], np, torch)
    resumed = list(saved['ledger']); steps(c, oc, saved['schedule'], saved['step'], 4, resumed)
    def same(x, y):
        if torch.is_tensor(x): return torch.equal(x, y)
        if isinstance(x, np.ndarray): return np.array_equal(x, y)
        if isinstance(x, dict): return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
        if isinstance(x, (tuple, list)): return len(x) == len(y) and all(same(p, q) for p, q in zip(x, y))
        return x == y
    checks = {'batch_dropout_loss_and_gradient_ledger_exact': whole == resumed,
        'AdamW_parameters_exact': same(expected['model'], c.state_dict()),
        'AdamW_optimizer_state_exact': same(expected['optimizer'], oc.state_dict()),
        'python_numpy_torch_RNG_exact': same(expected['rng'], capture_rng(np, torch))}
    assert all(checks.values()), checks
    report = {'status': 'PASS', 'mode': 'CPU toy continuous4 vs interruption2+restore2; same real-trainer helpers',
              'checks': checks, 'no_original_model_or_data_used': True, 'steps': 4}
    (folder / 'toy_receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--toy-check', required=True)
    print(json.dumps(toy_resume_check(ap.parse_args().toy_check)))

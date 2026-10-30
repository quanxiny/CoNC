"""Matched fixed-budget continuation with original trainability and static gates."""
from pathlib import Path
import argparse, collections, csv, fcntl, hashlib, json, os, sys, time
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from external_eval.boundary_optimization.common import (read, sha, digest, atomic_json, write_csv, setup, SourceData, load_model)
from external_eval.tifs.common import stable_id
from hybrid_retrain_core import ARMS, validate_gate, supervised_ce, combine_decision, selection_key
from hybrid_retrain_state import capture_rng, restore_rng, atomic_torch_save, repair_step_log
D = ROOT / 'external_eval/tifs_runs/hybrid_retraining_20261021'
INPUT_MAN = ROOT / 'external_eval/tifs_runs/tifs_mechanism_20261019/source_manifest.csv'


def validate_gate_identity(a, m, inp, source_registration):
    """Keep failed static analyses in supervision; bind each gate to original row."""
    assert type(a['row']) is int and a['row'] == m['row'] == int(inp['row'])
    assert a['unit_id'] == stable_id(m) == inp['sample_id']
    assert a['family_split'] == m['family_split'] == inp['historical_split']
    assert a['family_id'] == m['family_id'] == inp['family_id']
    assert str(a['source_case']) == str(m['source_case']) == str(inp['source_case'])
    assert a['consumed_input_sha256'] == inp['input_sha256'] and len(inp['input_sha256']) == 64
    assert not any(k in a for k in ['label', 'gold', 'N', 'neural_risk', 'neural_prediction'])
    g = validate_gate(a)
    assert str(source_registration['source_case']) == str(m['source_case'])
    assert source_registration['family_split'] == m['family_split']
    assert source_registration['family_id'] == m['family_id']
    assert len(source_registration['raw_source_sha256']) == 64 and source_registration['raw_source_path']
    registrations = [q for q in source_registration['queries'] if q['row'] == a['row']]
    assert len(registrations) == 1
    q = registrations[0]
    for key in ['unit_id', 'row', 'family_split', 'family_id', 'source_case', 'query_line', 'consumed_input_sha256']:
        assert q[key] == a[key], key
    if a['source_sha256'] is None:
        failure = 'NOT_COMPUTED_SOURCE_PREPARATION_FAILURE'
        assert a['D_calculation_status'] == a['Eva_calculation_status'] == failure
        assert a['source_path'] is None and g == {'D_trace': 'UNKNOWN', 'Eva': 'UNKNOWN', 'R': True}
    else:
        assert isinstance(a['source_sha256'], str) and len(a['source_sha256']) == 64
    return g


def parameter_manifest(model):
    return [{'name': n, 'shape': list(p.shape), 'dtype': str(p.dtype), 'trainable': p.requires_grad} for n, p in model.named_parameters()]


def frozen_parameters_sha(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_freeze():
    f = read(D / 'freeze.json')
    for n, h in f['files'].items(): assert sha(ROOT / n) == h, n
    assert f['config_sha256'] == sha(D / 'config.json')
    return f, read(D / 'config.json')


class GateData:
    def __init__(self, data, cfg):
        self.data = data; self.book = {}; self.rows = {}
        with INPUT_MAN.open() as stream:
            inputs = {int(r['row']): r for r in csv.DictReader(stream)}
        assert len(inputs) == len(data.metadata) == 117696
        source_manifest = read(D / 'gates/full/manifest.json')
        registrations = {str(s['source_case']): s for s in source_manifest['sources']}
        assert len(registrations) == source_manifest['source_count'] == 90151
        for split in ['train', 'val']:
            path = D / 'gates' / (split + '.jsonl'); expected = {i for i, m in enumerate(data.metadata) if m['family_split'] == split}
            got = set()
            with path.open() as stream:
                for line in stream:
                    a = json.loads(line); row = a['row']
                    assert type(row) is int and row in expected and row not in got
                    m = data.metadata[row]
                    g = validate_gate_identity(a, m, inputs[row], registrations[str(m['source_case'])])
                    self.book[row] = {**g, 'unit_id': a['unit_id']}; got.add(row)
            assert got == expected
            self.rows[split] = sorted(got)
        assert len(self.rows['train']) == 94179 and len(self.rows['val']) == 11940
        assert not set(self.rows['train']) & set(self.rows['val'])
        self.p_R = sum(self.book[i]['R'] for i in self.rows['train']) / len(self.rows['train'])
        self.mu = 1 + self.p_R
        assert abs(self.mu - cfg['mu']) < 1e-15
        import numpy as np
        for split in self.rows:
            rows = self.rows[split]
            assert np.array_equal(np.asarray(data.a['labels'][rows]), np.array([data.metadata[i]['label'] for i in rows]))


def count_metrics(y, prediction):
    tp = sum(a == 1 and b == 1 for a, b in zip(y, prediction)); fp = sum(a == 0 and b == 1 for a, b in zip(y, prediction))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, prediction)); tn = sum(a == 0 and b == 0 for a, b in zip(y, prediction))
    return {'n': len(y), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'precision': tp / (tp + fp) if tp + fp else None, 'recall': tp / (tp + fn) if tp + fn else None}


def gate_label_audit(gates):
    out = {}
    for split, rows in gates.rows.items():
        c = collections.Counter((gates.book[i]['R'], int(gates.data.a['labels'][i])) for i in rows)
        weighted_pos = sum((1 + int(r)) * c[r, 1] for r in [False, True]); weighted_total = sum((1 + int(r)) * c[r, y] for r in [False, True] for y in [0, 1])
        out[split] = {'n': len(rows), 'R0_label0': c[False, 0], 'R0_label1': c[False, 1],
            'R1_label0': c[True, 0], 'R1_label1': c[True, 1],
            'original_positive_fraction': sum(c[r, 1] for r in [False, True]) / len(rows),
            'weighted_positive_fraction': weighted_pos / weighted_total,
            'causal_limit': 'class-prior reweighting and gate-specific learning not separated without third arm'}
    rows = gates.rows['val']; truth = [int(gates.data.a['labels'][i]) for i in rows]
    static = [combine_decision(None, gates.book[i]['D_trace'], gates.book[i]['Eva'])['prediction'] for i in rows]
    ceiling = [y if gates.book[i]['R'] else p for i, y, p in zip(rows, truth, static)]
    nonR = [(y, p) for i, y, p in zip(rows, truth, static) if not gates.book[i]['R']]
    out['DEV_fixed_nonR'] = count_metrics([y for y, p in nonR], [p for y, p in nonR])
    out['DEV_perfect_R_hybrid_ceiling'] = count_metrics(truth, ceiling)
    out['mu'] = gates.mu; out['p_R'] = gates.p_R
    return out


def evaluate_dev(model, gates, device, torch):
    from torch.nn import functional as F
    model.eval(); records = []; ce = 0.
    with torch.inference_mode():
        for at in range(0, len(gates.rows['val']), 64):
            rows = gates.rows['val'][at:at + 64]; x, y = gates.data.batch(rows, device)
            logits = model(x)['logits']; ce += float(F.cross_entropy(logits, y, reduction='sum'))
            margin = logits[:, 1] - logits[:, 0]; risk = logits.softmax(-1)[:, 1]
            assert torch.isfinite(logits).all() and torch.isfinite(risk).all()
            for row, label, prob, score in zip(rows, y.cpu().tolist(), risk.cpu().tolist(), margin.cpu().tolist()):
                g = gates.book[row]; h = combine_decision(prob, g['D_trace'], g['Eva']); static = combine_decision(None, g['D_trace'], g['Eva'])
                assert h['residual'] == g['R']
                assert g['R'] or h['prediction'] == static['prediction']
                records.append({'row': row, 'unit_id': g['unit_id'], 'label': label,
                    'N': int(prob >= .5), 'risk': prob, 'margin': score, 'D_trace': g['D_trace'],
                    'Eva': g['Eva'], 'R': g['R'], 'H': h['prediction'], 'source': h['source']})
    y = [a['label'] for a in records]; rr = [a for a in records if a['R']]
    return {'N': count_metrics(y, [a['N'] for a in records]), 'H': count_metrics(y, [a['H'] for a in records]),
        'R_only': count_metrics([a['label'] for a in rr], [a['N'] for a in rr]),
        'NN_CE': ce / len(records)}, records


def job_identity(arm, seed, f, cfg):
    assert arm in ARMS and seed in cfg['seeds']
    ident = {'arm': arm, 'seed': seed, 'freeze_sha256': sha(D / 'freeze.json'),
        'config_sha256': sha(D / 'config.json'), 'base': f['base_models'][str(seed)],
        'gate_files': {s: sha(D / 'gates' / (s + '.jsonl')) for s in ['train', 'val']},
        'schedule_sha256': sha(D / 'schedules' / ('seed' + str(seed) + '.json')),
        'parameter_mask_sha256': f['parameter_contract']['mask_sha256'], 'mu': cfg['mu']}
    ident['run_id'] = digest(ident)
    return ident


def run(arm, seed, device):
    f, cfg = verify_freeze(); ident = job_identity(arm, seed, f, cfg)
    dest = D / 'training' / ident['run_id']; dest.mkdir(parents=True, exist_ok=True)
    lock = (dest / 'lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(dest / 'identity.json', ident, immutable=True)
    if (dest / 'completion.json').exists():
        r = read(dest / 'completion.json'); assert r['identity'] == ident
        for name in ['best', 'last']: assert sha(dest / (name + '.pt')) == r[name + '_sha256']
        print('VERIFIED_CACHED_TRAINING', arm, seed); return r
    np, torch = setup(seed); model, base = load_model(seed, device)
    assert base == ident['base']; assert digest(parameter_manifest(model)) == ident['parameter_mask_sha256']
    frozen_sha = frozen_parameters_sha(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    gates = GateData(SourceData(), cfg); audit = gate_label_audit(gates)
    atomic_json(dest / 'gate_label_and_ceiling_audit.json', audit, immutable=True)
    schedule = read(D / 'schedules' / ('seed' + str(seed) + '.json'))['rows']
    train_set = set(gates.rows['train'])
    assert len(schedule) == 1280 and all(len(a) == 32 and all(i in train_set for i in a) for a in schedule)
    state = {'step': 0, 'history': [], 'best_key': None, 'best_step': None, 'best_payload': None, 'elapsed_seconds': 0.}
    before = 0.; start = time.monotonic()
    if (dest / 'last.pt').exists():
        ck = torch.load(dest / 'last.pt', map_location='cpu', weights_only=False)
        assert ck['identity'] == ident and ck['frozen_parameters_sha256'] == frozen_sha
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); restore_rng(ck['rng'], np, torch, device)
        state = ck['training_state']; before = state['elapsed_seconds']
        # Last contains the committed selected payload, so an interrupted best
        # replacement cannot lose the preceding DEV-selected checkpoint.
        if state['best_payload'] is not None:
            atomic_torch_save(dest / 'best.pt', state['best_payload'], torch)
        del ck
    repair_step_log(dest / 'steps.jsonl', state['step'])
    rolling = hashlib.sha256()
    for batch in schedule[:state['step']]: rolling.update(np.asarray(batch, dtype='<i8').tobytes())
    def snapshot():
        state['elapsed_seconds'] = before + time.monotonic() - start
        return {'identity': ident, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'rng': capture_rng(np, torch, device), 'training_state': state,
            'frozen_parameters_sha256': frozen_sha, 'batch_rolling_sha256': rolling.hexdigest()}
    def validate():
        metrics, rows = evaluate_dev(model, gates, device, torch)
        assert metrics['H']['F1'] <= audit['DEV_perfect_R_hybrid_ceiling']['F1'] + 1e-12
        key = selection_key(metrics['H']['F1'], metrics['NN_CE']); chosen = state['best_key'] is None or key > tuple(state['best_key'])
        pred = dest / ('dev_predictions_step%04d.csv' % state['step'])
        if pred.exists():
            import csv
            with pred.open() as stream: old = list(csv.DictReader(stream))
            assert old == [{k: str(v) for k, v in row.items()} for row in rows], 'replayed DEV output differs'
        else: write_csv(pred, rows)
        if chosen:
            state['best_key'] = key; state['best_step'] = state['step']
            payload = {'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'identity': ident, 'step': state['step'], 'DEV': metrics, 'DEV_predictions': str(pred),
                'DEV_predictions_sha256': sha(pred), 'frozen_parameters_sha256': frozen_sha}
            state['best_payload'] = payload; atomic_torch_save(dest / 'best.pt', payload, torch)
        row = {'step': state['step'], 'validation_cycle': state['step'] // 128, 'selected': chosen,
            'metrics': metrics, 'elapsed_seconds': before + time.monotonic() - start}
        state['history'].append(row); atomic_json(dest / 'history.json', state['history'])
        print(json.dumps({'arm': arm, 'seed': seed, 'phase': 'DEV', **row}), flush=True)
    if not state['history']:
        validate(); atomic_torch_save(dest / 'last.pt', snapshot(), torch)
    with (dest / 'steps.jsonl').open('a') as ledger:
        while state['step'] < 1280:
            step = state['step']; ids = schedule[step]; model.train(); x, y = gates.data.batch(ids, device)
            residual = torch.tensor([gates.book[i]['R'] for i in ids], dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True); logits = model(x)['logits']
            loss, detail = supervised_ce(logits, y, residual, arm, gates.mu)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite supervised loss')
            loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'], error_if_nonfinite=True)
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            optimizer.step(); state['step'] += 1; rolling.update(np.asarray(ids, dtype='<i8').tobytes())
            ce = detail['CE'].detach(); r = detail['R']; record = {'step': state['step'],
                'loss': float(loss.detach()), 'NN_CE': float(ce.mean()), 'R_count': int(r.sum()),
                'R_CE': float(ce[r].mean()) if r.any() else None, 'nonR_CE': float(ce[~r].mean()) if (~r).any() else None,
                'grad_norm_before_clip': float(grad), 'batch_sha256': hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest(),
                'batch_rolling_sha256': rolling.hexdigest()}
            ledger.write(json.dumps(record) + '\n')
            if state['step'] % 64 == 0:
                ledger.flush(); os.fsync(ledger.fileno())
                if state['step'] % 128 == 0: validate()
                atomic_torch_save(dest / 'last.pt', snapshot(), torch)
                atomic_json(dest / 'progress.json', {**record, 'elapsed_seconds': before + time.monotonic() - start})
                print(json.dumps({'arm': arm, 'seed': seed, **record}), flush=True)
    assert frozen_parameters_sha(model) == frozen_sha
    best = state['best_payload']; assert best['step'] == state['best_step']
    result = {'status': 'COMPLETE_FIXED_1280_UPDATES', 'identity': ident, 'steps': state['step'],
        'validation_cycles': 10, 'validation_candidates_including0': len(state['history']),
        'best_step': best['step'], 'best_validation_cycle': best['step'] // 128, 'selected_DEV': best['DEV'],
        'last_DEV': state['history'][-1]['metrics'], 'actual_draws': 40960,
        'unique_sampled_TRAIN_rows': len({i for b in schedule for i in b}), 'TRAIN_equivalent_passes': 40960 / 94179,
        'elapsed_seconds': before + time.monotonic() - start, 'frozen_parameters_unchanged': True,
        'frozen_parameters_sha256': frozen_sha, 'best_sha256': sha(dest / 'best.pt'), 'last_sha256': sha(dest / 'last.pt'),
        'batch_rolling_sha256': rolling.hexdigest(), 'selection': 'same H-selected checkpoint also supplies N diagnostics; not N independently optimized',
        'outputs': {n: sha(dest / n) for n in ['history.json', 'steps.jsonl', 'gate_label_and_ceiling_audit.json']}}
    atomic_json(dest / 'completion.json', result, immutable=True); print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--arm', choices=ARMS, required=True); p.add_argument('--seed', type=int, required=True); p.add_argument('--device', default='cuda:0')
    a = p.parse_args(); run(a.arm, a.seed, a.device)

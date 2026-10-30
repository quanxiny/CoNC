"""Durable per-slice inference and strict all-required-slices scene aggregation."""
from __future__ import annotations
from collections import defaultdict
import hashlib,json,math,sqlite3
from pathlib import Path

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class PredictionStore:
    def __init__(self,path,contract):
        self.db=sqlite3.connect(path);self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS predictions(sample_id TEXT PRIMARY KEY,label INTEGER NOT NULL,status TEXT NOT NULL,logit REAL,probability REAL,prediction INTEGER,metadata TEXT NOT NULL)')
        current=json.dumps(contract,sort_keys=True,separators=(',',':'))
        old=self.db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
        if old is not None and old[0]!=current:self.db.close();raise ValueError('Prediction SQLite checkpoint/config/input hash mismatch')
        if old is None:
            if self.db.execute('SELECT count(*) FROM predictions').fetchone()[0]:raise ValueError('Uncontracted existing predictions')
            self.db.execute("INSERT INTO metadata VALUES('contract',?)",(current,));self.db.commit()
        self.contract=contract
    def ids(self):return {r[0] for r in self.db.execute('SELECT sample_id FROM predictions')}
    def write(self,rows):
        with self.db:
            self.db.executemany('INSERT INTO predictions VALUES(?,?,?,?,?,?,?)',[
                (r['sample_id'],int(r['label']),r['status'],r.get('logit'),r.get('probability'),r.get('prediction'),json.dumps(r,sort_keys=True,separators=(',',':'))) for r in rows])
    def rows(self):return [json.loads(row[0]) for row in self.db.execute('SELECT metadata FROM predictions ORDER BY sample_id')]
    def close(self):self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)');self.db.close()

def score_missing(store,records,expected,infer,batch_size=256,after_batch=None):
    """Commit after each successful batch; an interrupted batch alone may repeat."""
    available={r['meta']['sample_id']:r for r in records};known=store.ids();expected_ids={r['sample_id'] for r in expected}
    if len(expected_ids)!=len(expected):raise ValueError('Duplicate required sample identity')
    if known-expected_ids:raise ValueError('Prediction rows not in expected population')
    uncovered=[]
    for r in expected:
        if r['sample_id'] not in known and r['sample_id'] not in available:
            uncovered.append({**r,'status':'uncovered','coverage':False,'logit':None,'probability':None,'prediction':None})
    if uncovered:store.write(uncovered)
    remaining=[r for r in expected if r['sample_id'] not in known and r['sample_id'] in available]
    inferred=0
    for start in range(0,len(remaining),batch_size):
        meta=remaining[start:start+batch_size];logits,probabilities=infer([available[r['sample_id']] for r in meta])
        if len(logits)!=len(meta) or len(probabilities)!=len(meta):raise ValueError('Model output batch size mismatch')
        rows=[]
        for r,logit,p in zip(meta,logits,probabilities):
            if not math.isfinite(logit) or not math.isfinite(p):raise ValueError('Nonfinite model output')
            rows.append({**r,'status':'scored','coverage':True,'logit':float(logit),'probability':float(p),'prediction':int(p>.5)})
        store.write(rows);inferred+=len(rows)
        if after_batch:after_batch(inferred)
    return inferred

def aggregate(expected,scored):
    lookup={r['sample_id']:r for r in scored};groups=defaultdict(list)
    for r in expected:groups[r['scene_id']].append(r)
    results=[]
    for scene,rows in sorted(groups.items()):
        if len({r['label'] for r in rows})!=1:raise ValueError('Scene labels disagree')
        required=sorted(r['sample_id'] for r in rows);complete=[lookup[s] for s in required if s in lookup and lookup[s]['status']=='scored']
        covered=len(complete)==len(required)
        # required IDs and complete rows are ascending; max keeps the first probability tie.
        winner=max(complete,key=lambda r:r['probability']) if covered else None
        base={k:v for k,v in rows[0].items() if k not in {'sample_id','source_lines','source_case','key_source_line','coverage','coverage_status'}}
        results.append({**base,'scene_id':scene,'coverage':covered,'coverage_status':'all_required_slices_scored' if covered else 'UNKNOWN_incomplete_required_slices',
                        'required_slice_ids':required,'expected_slices':len(required),'covered_slices':len(complete),
                        'missing_slice_ids':[s for s in required if s not in lookup or lookup[s]['status']!='scored'],
                        'max_risk_slice_id':winner['sample_id'] if winner else None,'logit':winner['logit'] if winner else None,
                        'probability':winner['probability'] if winner else None,'prediction':winner['prediction'] if winner else None})
    return results

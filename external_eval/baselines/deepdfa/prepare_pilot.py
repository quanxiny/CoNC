"""Export the fixed 64 TRAIN cases for a native static Joern cost pilot.

Source is never executed. Frozen original slice line numbers are retained;
comments are blanked without changing line/column positions or literals.
"""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'external_eval/paper_baselines/deepdfa/native_pilot'
DATA=ROOT/'datasets/overflowgen_24455320/data/overflow_gen'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def remove_comments(text):
    pattern=re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',re.S|re.M)
    return pattern.sub(lambda m: ''.join('\n' if c=='\n' else ' ' for c in m.group())
                       if m.group().startswith('/') else m.group(),text)

def main():
    check='char *url="http://x"; /* comment\n still comment */ int a=1; // Tag.BAD\n'
    cleaned=remove_comments(check)
    assert 'http://x' in cleaned and 'Tag.BAD' not in cleaned and cleaned.count('\n')==check.count('\n')
    census=json.loads((OUT.parent/'schema_census.json').read_text())
    selected=set(census['selected_source_cases'])
    manifest=ROOT/'transformer_gru_v76_v2/external/overflowgen/joint_graph_study_v1/sample_ids.jsonl.gz'
    assert sha(manifest)==census['manifest_sha256']
    with gzip.open(manifest,'rt') as f:rows=[json.loads(s) for s in f]
    selected_rows=[r for r in rows if r['source_case'] in selected]
    assert all(r['family_split']=='train' for r in selected_rows)
    originals={s:json.loads((DATA/f'{s}_data.json').read_text()) for s in {'train','val','test'}}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'source').mkdir(exist_ok=True)
    records=[]
    for case in sorted(selected):
        matches=[r for r in selected_rows if r['source_case']==case]
        original=originals[matches[0]['original_split']][matches[0]['original_sample_id']]
        rel=Path(original['file_path']).relative_to('data/overflow_gen')
        source=DATA/rel
        raw=source.read_text()
        cleaned=remove_comments(raw)
        assert len(raw)==len(cleaned) and raw.count('\n')==cleaned.count('\n')
        target=OUT/'source'/f'{case}.c'
        target.write_text(cleaned)
        slices=[]
        for row in matches:
            item=originals[row['original_split']][row['original_sample_id']]
            assert item['file_path']==original['file_path']
            slices.append({'row':row['row'],'original_split':row['original_split'],
                           'original_sample_id':row['original_sample_id'],'family_split':'train',
                           'source_line_numbers':item['line_num'],'label':int(item['label'])})
        records.append({'source_case':case,'source_original':str(source.relative_to(ROOT)),
                        'source_original_sha256':sha(source),'source_clean':str(target.relative_to(ROOT)),
                        'source_clean_sha256':sha(target),'slices':slices})
    receipt={'status':'PREPARED_TRAIN_ONLY_NOT_PARSED','cases':len(records),'slices':sum(len(r['slices']) for r in records),
             'source_manifest_sha256':sha(manifest),'schema_census_sha256':sha(OUT.parent/'schema_census.json'),
             'adapter_sha256':sha(Path(__file__)),'comment_policy':'blank comments, preserve literals, every byte position and line number for ASCII source',
             'prediction_unit':'native CFG induced on original frozen slice lines; evaluate this adaptation only after native feature coverage acceptance',
             'source_executed':False,'records':records}
    (OUT/'source_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='records'}))

if __name__=='__main__':main()

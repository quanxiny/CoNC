"""Official feature functions on native Joern CPGs; matched-slice projection.

Selected upstream functions are compiled without changing their bodies, avoiding
unrelated BigVul download and CLI dependencies. No legacy graph conversion exists.
"""
from __future__ import annotations
import ast
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import traceback
import types
import time
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sparse

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OFFICIAL=ROOT/'transformer_gru_v76_v2/third_party/DeepDFA/DDFA'
OUT=ROOT/'external_eval/paper_baselines/deepdfa/native_pilot'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def functions(path,names,namespace,assignment_names=()):
    source=path.read_text(); tree=ast.parse(source)
    selected=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:selected.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in assignment_names for t in node.targets):selected.append(node)
    if len([n for n in selected if isinstance(n,ast.FunctionDef)])!=len(names):raise ValueError('Missing upstream function')
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return {name:namespace[name] for name in names}

EXPECTED_SOURCES = {'transformer_gru_v76_v2/third_party/DeepDFA/DDFA/sastvd/helpers/joern.py': '7153c711a7b7596586d9509afa4fb0488165014d73d122e5c4c031a989970970', 'transformer_gru_v76_v2/third_party/DeepDFA/DDFA/code_gnn/analysis/dataflow.py': 'd9f845405e8c3c2509738789e0fce63a1a69df4c04d70ba52f8ea233471565ca', 'transformer_gru_v76_v2/third_party/DeepDFA/DDFA/sastvd/scripts/abstract_dataflow_full.py': 'e9633fd8e6f3c048708a8b1edb153b60a61398d9912ff02d366ae82b387f7c14', 'transformer_gru_v76_v2/third_party/DeepDFA/DDFA/sastvd/linevd/utils.py': '4cedd8106577c5f41b592f11b91980bd0bdb862e29ebf26fbc678c65caefd47b'}

def extract(manifest):
    for relative, expected in EXPECTED_SOURCES.items():
        if sha(ROOT/relative) != expected: raise ValueError("Official feature source changed: "+relative)
    paths={r['source_case']:ROOT/r['source_clean'] for r in manifest['records']}
    helper_file=OFFICIAL/'sastvd/helpers/joern.py'
    helper_ns={'Path':Path,'json':json,'np':np,'pd':pd,'defaultdict':defaultdict,'sparse':sparse}
    helpers=functions(helper_file,['get_node_edges','neighbour_nodes','rdg','assign_line_num_to_local','drop_lone_nodes'],helper_ns)
    # Both unchanged upstream stages read the same graph. Cache their common
    # loader result, returning deep copies because upstream stages mutate frames.
    original_loader=helpers['get_node_edges']; frame_cache={}
    def cached_loader(filepath,verbose=0):
        cache_key=str(filepath)
        if cache_key not in frame_cache:frame_cache[cache_key]=original_loader(filepath,verbose=verbose)
        return tuple(frame.copy(deep=True) for frame in frame_cache[cache_key])
    helpers['get_node_edges']=cached_loader
    helper=types.SimpleNamespace(**helpers)
    dataflow_ns={'nx':nx,'svdj':helper,'svdds':types.SimpleNamespace(itempath=lambda ident,dsname:paths[ident])}
    dataflow=types.SimpleNamespace(**functions(OFFICIAL/'code_gnn/analysis/dataflow.py',['sub','get_cpg'],dataflow_ns))
    feature_ns={'nx':nx,'pd':pd,'re':re,'json':json,'traceback':traceback,'dataflow':dataflow,'dsname':'overflowgen'}
    native=functions(OFFICIAL/'sastvd/scripts/abstract_dataflow_full.py',['is_decl','get_dataflow_features','cleanup_datatype','to_hash'],feature_ns,['all_assignment_types'])
    cfg_ns={'pd':pd,'svdj':helper}
    cfg=functions(OFFICIAL/'sastvd/linevd/utils.py',['get_iddict','feature_extraction'],cfg_ns)
    graphs=[]; cases=[]
    for record in manifest['records']:
        case=record['source_case']; path=paths[case]
        if not Path(str(path)+'.nodes.json').is_file():raise FileNotFoundError(path)
        if sha(path)!=record['source_clean_sha256']:raise ValueError('Pilot source changed')
        entry={'source_case':case,'source_clean_sha256':sha(path)}
        started=time.monotonic()
        try:
            raw_nodes=json.loads(Path(str(path)+'.nodes.json').read_text())
            raw_edges=json.loads(Path(str(path)+'.edges.json').read_text())
            source_methods=[r['id'] for r in raw_nodes if r.get('_label')=='METHOD'
                            and r.get('filename')==str(path) and not r.get('isExternal',False)]
            if not source_methods:raise ValueError('No native method belongs to the source file')
            ast_graph=nx.DiGraph((e[1],e[0]) for e in raw_edges if e[2]=='AST')
            source_owned=set(source_methods)
            for method in source_methods:source_owned.update(nx.descendants(ast_graph,method))
            # Upstream get_dataflow_features_df concatenates with ignore_index=True
            # before cleanup_datatype; reproduce that index reset for one case.
            df=native['get_dataflow_features'](case,raise_all=True).reset_index(drop=True)
            feature_ns['dataflow_df']=df
            native['cleanup_datatype'](df)
            hashes={int(nid):json.loads(native['to_hash'](group,['api','datatype','literal','operator']))
                    for nid,group in df.groupby('node_id')}
            nodes,edges=cfg['feature_extraction'](path,graph_type='cfg',group=False)
            entry.update(status='native_extracted',cfg_nodes=len(nodes),cfg_edges=len(edges),definition_nodes=len(hashes),
                         feature_counts={k:int(v) for k,v in df['subkey'].value_counts().items()},slice_count=len(record['slices']))
            for original in record['slices']:
                same_lines=nodes[nodes.lineNumber.isin(original['source_line_numbers'])]
                selected=same_lines[same_lines.id.isin(source_owned)].copy()
                index={int(old):i for i,old in enumerate(selected.index)}
                internal=edges[edges.innode.isin(index)&edges.outnode.isin(index)]
                node_records=[{'native_id':int(row.id),'line':int(row.lineNumber),
                               'label':row._label,'code':row.code,'features':hashes.get(int(row.id))}
                              for _,row in selected.iterrows()]
                # Upstream dbize_graphs.py uses innode -> outnode, then self loops.
                selected_edges=[[index[int(e.innode)],index[int(e.outnode)]] for _,e in internal.iterrows()]
                definition_count=sum(r['features'] is not None for r in node_records)
                status='accepted' if node_records else 'no_native_cfg_in_frozen_slice'
                graphs.append({'row':original['row'],'family_split':original.get('split',original.get('family_split','train')),'label':original['label'],
                               'source_case':case,'source_line_numbers':original['source_line_numbers'],
                               'status':status,'nodes':node_records,'edges':selected_edges,
                               'definition_count':definition_count,'add_self_loops':True,
                               'other_file_line_collisions_excluded':len(same_lines)-len(selected)})
        except Exception as error:
            entry.update(status='feature_extraction_failed',error=repr(error),traceback=traceback.format_exc())
        entry['elapsed_seconds']=time.monotonic()-started
        cases.append(entry)
    files=[helper_file,OFFICIAL/'code_gnn/analysis/dataflow.py',OFFICIAL/'sastvd/scripts/abstract_dataflow_full.py',OFFICIAL/'sastvd/linevd/utils.py']
    return {'status':'NATIVE_FEATURE_PILOT','official_sources':{str(p.relative_to(ROOT)):sha(p) for p in files},
            'source_manifest_sha256':manifest.get('source_manifest_sha256',sha(OUT/'source_manifest.json')),'adapter_sha256':sha(Path(__file__)),
            'case_status':dict(Counter(r['status'] for r in cases)),'slice_status':dict(Counter(r['status'] for r in graphs)),
            'cases':cases,'graphs':graphs,'feature_body_changes':False,
            'adaptation':'Induced native CFG over frozen slice lines, including isolated retained CFG vertices with official self loops; full source is used only by the native parser/type resolver.',
            'model_training':False,'model_predictions':False}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=64)
    args=parser.parse_args()
    manifest=json.loads((OUT/'source_manifest.json').read_text())
    manifest['records']=manifest['records'][:args.limit]
    result=extract(manifest)
    destination=OUT/'native_features.json'
    if destination.exists():
        previous_sha=sha(destination)
        (OUT/f'native_features_prior_{previous_sha[:12]}.json').write_bytes(destination.read_bytes())
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','graphs']}))

if __name__=='__main__':main()

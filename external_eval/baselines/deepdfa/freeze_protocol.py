"""One-time pretraining protocol freeze, independent of held-out predictions."""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'external_eval/paper_baselines/deepdfa'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=OUT/'protocol.json'
    x={'version':1,'method':'DeepDFA official GGNN matched-slice native-CFG adaptation',
       'source':str(ROOT/'transformer_gru_v76_v2/third_party/DeepDFA'),'family_slices':{'train':94179,'val':11940,'test':11577},
       'seeds':[11,29,47],'model':{'input_dim':1002,'hidden_dim_per_feature':32,'concat_four_features':True,'n_steps':5,'num_output_layers':3},
       'training':{'epochs':25,'batch_size':256,'optimizer':'Adam','lr':.001,'weight_decay':.01,'equal_negative_subsampling_each_epoch':True,
                   'selection':'minimum full covered source DEV binary cross entropy','threshold':'sigmoid(logit) > 0.5','hyperparameter_grid':False},
       'vocabulary':{'fit':'TRAIN only, labels not inspected','atomic_top_k':1000,'composite_top_k':1000,'nondefinition':0,'oov':1,'known':'2..1001',
                     'tie_break':'count descending then canonical JSON lexicographic','identity':'native graph_id + node_id; count each unique native definition once across overlapping slices',
                     'adaptation':'Fix apparent upstream dataframe index alignment by explicitly mapping each graph/node key; independently fitted atomic+composite per api/datatype/literal/operator'},
       'frontend':{'parser':'official Joern1.1.1072 full original sanitized source, native CFG induced on frozen slice lines',
                   'source_ownership':'nonexternal METHOD exact source filename and AST descendants','edge_orientation':'official innode -> outnode; explicit self loops',
                   'isolates':'selected native CFG nodes retained after slice induction','coverage':'at least one native CFG node; nondefinitions explicitly 0 as official dbize_absdf.py',
                   'no_definition_graphs':'Retain per official graphmogrifier.py/dataset.py; record definition_count=0',
                   'context_limitation':'Full source parsing gives type/context beyond frozen slice; this is not original full-function benchmark reproduction',
                   'timeouts_seconds':90,'failure_policy':'Explicit error ledger; unknown inputs never relabelled safe; retry only incomplete cases interrupted behind timeout'},
       'evaluation':{'source_test':'held out until DEV checkpoint frozen','external':'MIT and FormAI only frozen transfer; max probability over covered slices per scene',
                     'coverage':'report native coverage and existing frozen common intersection; compare all methods on identical available cohort',
                     'no_target_tuning':True},
       'official_model_sha256':sha(HERE/'native_model.py'),'official_model_regression_sha256':sha(OUT/'model_acceptance.json'),
       'old_experiments':'No original predictions or checkpoints modified; no legacy fusion branch',
       'status':'FROZEN_BEFORE_FORMAL_TRAINING'}
    encoded=json.dumps(x,indent=2)+'\n'
    if p.exists():
        if p.read_text()!=encoded:raise ValueError('Protocol already frozen with different content')
    else:p.write_text(encoded)
    print(sha(p))
if __name__=='__main__':main()

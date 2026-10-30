"""Full-capacity bridges; no missing-pretraining fallback and no analyzer launch."""
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
from hashlib import sha256
import importlib.util,json,math
import numpy as np
import torch
from torch import nn

OVERFLOW_ROOT=Path((('${OVERFLOWGEN_ROOT}').replace('${OVERFLOWGEN_ROOT}',__import__('os').environ.get('OVERFLOWGEN_ROOT',str(__import__('pathlib').Path(__file__).resolve().parents[4]/'external/overflowgen')))))
AUTHOR_LAYERS_SHA='f4fc164f32e462260824f31a9aa956218743bfacb84a912285709c25bc950e7c'

class NativeVD(nn.Module):
    """Author 512/3-local/2-global/direct-head tensor code, original DICE and W2V.

    family track random word initialization is an explicit adaptation because
    original W2V was trained on the author's different train partition.
    """
    def __init__(self,track='author',root=OVERFLOW_ROOT):
        super().__init__()
        if track not in {'author','family'}:raise ValueError('unknown track')
        root=Path(root);path=root/'author_code/vdtransformer/src/models/modules/transformer_viz.py'
        data=path.read_bytes()
        if sha256(data).hexdigest()!=AUTHOR_LAYERS_SHA:raise ValueError('author model code changed')
        ns={};source=data.decode().replace('from omegaconf import DictConfig','DictConfig = object')
        exec(compile(source,str(path),'exec'),ns) # annotation-only dependency substitution
        words=json.loads((root/'safe_numeric_artifacts/word_index.json').read_text())
        self.vocabulary={w:i+3 for i,w in enumerate(words)}|{'<PAD>':0,'<UNK>':1,'<MASK>':2}
        self.embedding=nn.Embedding(len(self.vocabulary),512,padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)
        if track=='author':
            with torch.no_grad():self.embedding.weight[3:].copy_(torch.from_numpy(np.load(root/'safe_numeric_artifacts/word_vectors.npy',allow_pickle=False)))
        q=torch.from_numpy(np.load(root/'safe_numeric_artifacts/dice_Q.npy',allow_pickle=False)).float()
        theta=torch.arange(1000)/1000*math.pi
        polar=torch.stack([theta.sin()**(d-1)*theta.cos() if d<512 else theta.sin()**512 for d in range(1,513)],-1)
        self.register_buffer('number_table',torch.einsum('ik,nk->ni',q,polar))
        cfg=SimpleNamespace(embed_size=512,
            l_transformer=SimpleNamespace(pe_dropout=0.,nhead=4,dim_feedforward=2048,dropout=.1,number_layers=3),
            g_transformer=SimpleNamespace(pe_dropout=0.,nhead=2,dim_feedforward=4096,dropout=.1,number_layers=2,num_class=2))
        self.local=ns['TransformerEncoderL'](cfg);self.global_encoder=ns['TransformerEncoderG_Formatter'](cfg)
        self.track=track
    def forward(self,*,token_ids,number_ids,number_mark,line_counts,key_padding_mask,key_lines):
        if number_ids.dtype!=torch.long:raise TypeError('VD numbers must stay integer before DICE lookup')
        if ((number_ids<0)|(number_ids>=1000)).any():raise ValueError('outside audited OverflowGen DICE domain; no clamp')
        emb=self.embedding(token_ids)*number_mark.unsqueeze(-1)+self.number_table[number_ids]*(1-number_mark).unsqueeze(-1)
        sentences,_=self.local(emb,token_ids.eq(0),line_counts,key_padding_mask)
        logits,_=self.global_encoder(sentences,key_padding_mask,key_lines)
        return logits

class FullRoberta(nn.Module):
    """Active published RoBERTa path. Baseline head is two logits (LineVul) or
    one logit (PrimeVul UniXcoder). Structural fixture mode is random init only.
    The author's unused duplicate superclass backbone is not allocated here;
    active architecture is unchanged but checkpoint keys require explicit mapping.
    """
    def __init__(self,kind,*,local_weights=None,structural_fixture=False):
        super().__init__()
        from transformers import RobertaConfig,RobertaForSequenceClassification
        if kind not in {'linevul','unixcoder'}:raise ValueError(kind)
        labels=2 if kind=='linevul' else 1
        if structural_fixture:
            if local_weights is not None:raise ValueError('fixture mode cannot claim pretrained weights')
            cfg=RobertaConfig(vocab_size=50265,hidden_size=768,num_hidden_layers=12,num_attention_heads=12,
                intermediate_size=3072,max_position_embeddings=514,pad_token_id=1,num_labels=labels)
            self.encoder=RobertaForSequenceClassification(cfg)
        else:
            if local_weights is None or not Path(local_weights).is_dir():raise FileNotFoundError('full local pretrained checkpoint required; no download or random fallback')
            self.encoder=RobertaForSequenceClassification.from_pretrained(local_weights,local_files_only=True,num_labels=labels)
            cfg=self.encoder.config
            if (cfg.hidden_size,cfg.num_hidden_layers,cfg.num_attention_heads,cfg.intermediate_size)!=(768,12,12,3072):
                raise ValueError('full original backbone required')
        self.kind=kind;self.structural_fixture=structural_fixture
    def forward(self,input_ids):
        return self.encoder(input_ids=input_ids,attention_mask=input_ids.ne(1)).logits

class AuthorFusionHead(nn.Module):
    """DeepDFA author's fusion HEAD equations, not a replacement GGNN model.
    Full execution additionally requires the original flowgnn encoder and graphs.
    """
    def __init__(self,flow_dim,dropout=.1):
        super().__init__();self.dense=nn.Linear(768+flow_dim,768);self.dropout=nn.Dropout(dropout);self.out_proj=nn.Linear(768,2)
    def forward(self,token_features,flow):
        x=torch.cat([token_features[:,0,:],flow],-1)
        return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

def encode_text(text,tokenizer,max_tokens=512):
    # Both checked published training recipes use plain CLS/code/SEP, including
    # PrimeVul UniXcoder. Native UniXcoder <encoder-only> is a separate adaptation.
    ids=tokenizer.convert_tokens_to_ids([tokenizer.cls_token]+tokenizer.tokenize(text)[:max_tokens-2]+[tokenizer.sep_token])
    return ids+[tokenizer.pad_token_id]*(max_tokens-len(ids))

@dataclass(frozen=True)
class AuthorJob:
    method:str
    argv:tuple[str,...]
    cwd:str
    manifest_sha256:str
    blockers:tuple[str,...]

def deepdfa_job(author_root,manifest_sha256,*,fusion=False):
    root=Path(author_root)
    if fusion:
        argv=('python','linevul_main.py','--model_type=roberta','--model_name_or_path=microsoft/codebert-base',
              '--tokenizer_name=microsoft/codebert-base','--do_train','--epochs=10','--block_size=512',
              '--train_batch_size=16','--eval_batch_size=16','--learning_rate=2e-5','--max_grad_norm=1')
        cwd=root/'LineVul/linevul';name='author_LineVul_DeepDFA_fusion'
    else:
        argv=('python','code_gnn/main_cli.py','fit','--config','configs/config_bigvul.yaml','--config','configs/config_ggnn.yaml')
        cwd=root/'DDFA';name='author_DeepDFA'
    return AuthorJob(name,argv,str(cwd),manifest_sha256,
        ('original repository/config commit not locally pinned','DGL/Lightning absent','approved original CFG/abstract-dataflow features absent',
         'shared split paths and graph row IDs must replace BigVul defaults before execution',
         'Joern README 1.1.1072 vs installer 1.1.107 unresolved; analyzer execution unavailable'))

def execute_author_job(job):
    # A dry-run spec must never silently launch the original default dataset or its --do_test.
    raise RuntimeError('author job is draft only: '+ '; '.join(job.blockers))

class LegacyPureGraph(nn.Module):
    """Unmodified FULL previously compared graph baseline; not v51's small ablation.
    Semantic includes the original digit GRU; numeric has all public candidate bits
    and structure. Neither forward receives sample labels or binding truth.
    """
    def __init__(self,arm):
        super().__init__()
        if arm not in {'semantic','numeric','joint'}:raise ValueError(arm)
        path=OVERFLOW_ROOT/'joint_graph_study_v1/implementation_family_isolated_v2/models.py'
        data=path.read_bytes()
        if sha256(data).hexdigest()!='cfeef58709837e62ef24ea9d882250d9272aaaad4b71f7f2c1d7f98e9ec89101':raise ValueError('legacy baseline source changed')
        ns={};exec(compile(data,str(path),'exec'),ns)
        self.input_keys=ns['INPUT_KEYS'];self.model=ns['IndependentGraphModel'](arm,width=576 if arm=='numeric' else 512,layers=4)
    def forward(self,**x):
        if set(x)!=set(self.input_keys):raise ValueError('closed baseline input schema')
        return self.model(**x)

class RiskBridge(nn.Module):
    """Two-logit CE and binary BCE are identical at unit class weights."""
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):
        logits=self.model(**x);p=logits.softmax(-1)[:,1]
        return {'risk':p,'lower':p,'upper':p,'accepted':torch.ones_like(p,dtype=torch.bool),'logits':logits}

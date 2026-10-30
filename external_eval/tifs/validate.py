"""Finite real-model regression and intervention invariants before queue freeze."""
import copy,gc,time
from .common import *

def run(out,device='cpu'):
    np,torch=setup(1234)
    from .models import make_model,ControlledQLayer,initial_q,transfer_stage1
    from all_visible52 import AllVisible
    from model70 import AdaptiveGraph70
    from overflow51.baselines import RiskBridge
    from overflow51.training import TrainConfig,Cursor,EarlyStop,snapshot,restore
    from torch.nn import functional as F
    data=SourceData();train=[i for i,r in enumerate(data.metadata) if r['family_split']=='train'][:4]
    x,y=data.batch(train,device);checks={};counts={};start=time.monotonic()
    vd=make_model('vd',1,1234).to(device).train();xx,yy=data.batch(train,device,vd=True)
    vv=vd(xx)['logits'];assert vv.shape==(len(train),2) and vv.isfinite().all();F.cross_entropy(vv,yy).backward()
    checks['native_VD_original_batch_forward_backward']=True;del vd,xx,yy,vv;gc.collect()
    for stage in [1,2]:
        torch.manual_seed(1234);original=RiskBridge(AllVisible()) if stage==1 else AdaptiveGraph70('liquid_transfer_frozen')
        torch.manual_seed(1234);wrapper=make_model('full',stage,1234)
        assert set(original.state_dict())==set(wrapper.state_dict())
        assert all(torch.equal(original.state_dict()[k],wrapper.state_dict()[k]) for k in original.state_dict())
        original=original.eval().to(device);wrapper=wrapper.eval().to(device)
        with torch.no_grad():a=original(x)['logits'];b=wrapper(x)['logits']
        assert torch.equal(a,b),'CoNC no-op output drift'
        checks['full_stage'+str(stage)+'_bitwise_original_parity']=True
        del original,wrapper;gc.collect()
    h=torch.randn(2,5,448,device=device);q=torch.rand(2,5,36,device=device)
    parents=torch.tensor([[[0,0],[0,0],[1,0],[1,2],[2,1]]]*2,device=device)
    ops=torch.tensor([[12,1,1,3,4]]*2,device=device);bits=torch.randint(0,2,(2,5,32),device=device).float()
    known=torch.tensor([[0,1,1,0,0]]*2,device=device).float()
    for variant in config()['variants']:
        torch.manual_seed(1234);m=make_model(variant,1,1234).to(device).eval();layer=m.model.q_layers[0]
        counts[variant]={'stage1_total':sum(p.numel() for p in m.parameters()),'stage1_requires_grad':sum(p.numel() for p in m.parameters() if p.requires_grad)}
        counts[variant]['operator_parameters']=sum(p.numel() for name,p in layer.named_parameters() if name.startswith(('bit_rnn.','bit_out.','bit_mlp.')))
        counts[variant]['operator_proposal_calls_per_Q_layer']=2 if variant=='symmetric_q' else 1
        with torch.no_grad():
            for _ in range(5):layer(h,q,parents,ops,bits,known)
            if device.startswith('cuda'):torch.cuda.synchronize()
            began=time.perf_counter()
            for _ in range(30):layer(h,q,parents,ops,bits,known)
            if device.startswith('cuda'):torch.cuda.synchronize()
            counts[variant]['Q_layer_forward_ms_fixture_B2_N5']=1000*(time.perf_counter()-began)/30
            output=layer(h,q,parents,ops,bits,known)
            assert output.shape==q.shape and output.isfinite().all()
            if variant!='no_clamp':
                literal=torch.cat([bits,torch.ones_like(known[...,None]),torch.zeros_like(bits[...,:3])],-1)
                assert torch.equal(output[known.bool()],literal[known.bool()])
            else:assert not torch.equal(output[known.bool()][...,:32],bits[known.bool()])
            if variant in ('no_h_to_q','no_both_dynamic'):assert torch.equal(output,layer(h*5+7,q,parents,ops,bits,known))
            if variant=='local_q':assert torch.equal(output,layer(h,q,parents.flip(-1),ops,bits,known))
            if variant=='symmetric_q':assert torch.equal(output,layer(h,q,parents.flip(-1),ops,bits,known))
        m.train();result=m(x);loss=F.cross_entropy(result['logits'],y);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or p.grad.isfinite().all() for p in m.parameters())
        counts[variant]['fixture_nonzero_gradient_parameters']=sum(int(p.grad.ne(0).sum()) for p in m.parameters() if p.grad is not None)
        checks[variant+'_forward_backward_and_literal_contract']=True
        stage2=make_model(variant,2,1234).to(device);transfer_stage1(stage2,m.state_dict());stage2.eval();m.eval()
        with torch.no_grad():assert torch.equal(m(x)['logits'],stage2(x)['logits']),'Stage2 transfer drift: '+variant
        counts[variant]['stage2_total']=sum(p.numel() for p in stage2.parameters());counts[variant]['stage2_requires_grad']=sum(p.numel() for p in stage2.parameters() if p.requires_grad)
        checks[variant+'_stage2_transfer_and_zero_residual_parity']=True
        del m,layer,stage2;gc.collect()
        if device.startswith('cuda'):torch.cuda.empty_cache()
    for variant in ['no_dynamic_q_to_h','no_both_dynamic']:
        for stage in [1,2]:
            torch.manual_seed(1234);m=make_model(variant,stage,1234).eval().to(device)
            if stage==2:
                with torch.no_grad():m.residual_scale.fill_(.3) # exercise refiner Q path even when original beta starts at zero
            backbone=m.model if stage==1 else m.backbone
            with torch.no_grad():a=m(x)['logits']
            handles=[layer.register_forward_hook(lambda mod,args,res:torch.full_like(res,.123)) for layer in backbone.q_layers]
            with torch.no_grad():b=m(x)['logits']
            for handle in handles:handle.remove()
            assert torch.equal(a,b),'Uncut dynamic Q route in '+variant
            checks[variant+'_stage'+str(stage)+'_all_dynamic_paths_isolated']=True
            del m;gc.collect()
    # Resume actual full backbone with dropout, optimizer and sampler RNG.
    torch.manual_seed(1234);m=make_model('full',1,1234).to(device).train()
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4);sched=torch.optim.lr_scheduler.ExponentialLR(opt,.95)
    cursor=Cursor(len(train),1234);stop=EarlyStop();tc=TrainConfig(seed=1234)
    def step():
        ix=cursor.next(2);xx,yy=data.batch([train[int(i)] for i in ix],device)
        opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m(xx)['logits'],yy);loss.backward();opt.step();return float(loss.detach())
    step();saved=snapshot(m,opt,sched,cursor,stop,tc,'fixture',1);a=step();after={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    restore(saved,m,opt,sched,cursor,stop,tc,'fixture');b=step()
    assert a==b and all(torch.equal(after[k],v.detach().cpu()) for k,v in m.state_dict().items())
    checks['actual_backbone_exact_optimizer_cursor_rng_resume']=True
    del m,opt,saved,after;gc.collect()
    # Independent fixtures for conservative tensor rules and aggregation assumptions.
    from .rules import decide
    xx={k:np.array(v[0].cpu()) for k,v in x.items()};xx['node_count']=np.array(5);xx['ops']=np.array([12,1,1,3,16]);xx['parents']=np.array([[0,0],[0,0],[0,0],[1,2],[3,2]])
    xx['known']=np.array([0,1,1,0,0]);xx['bits']=np.zeros((5,32));xx['bits'][1,0]=1;xx['bits'][2,3]=1;xx['target']=np.array([0,0,0,0,1]);xx['condition']=np.ones(5);xx['unit']=np.ones(5)
    assert decide(xx)['prediction']==1
    xx['ops'][3]=4;assert decide(xx)['prediction']==1 # 1-8 negative index
    xx['parents'][4]=[1,2];assert decide(xx)['prediction']==0
    xx['parents'][4]=[0,2];assert decide(xx)['prediction'] is None
    xx['parents'][4]=[1,2];xx['condition'][4]=2;assert decide(xx)['prediction'] is None
    checks['rule_negative_index_known_safe_missing_operand_guard_unknown']=True
    result={'status':'PASS','checks':checks,'parameter_counts':counts,'train_fixture_rows':train,'device':device,
            'wall_s':time.monotonic()-start,'trained_fixture_weights_discarded':True,'formal_training_started':False,
            'code':code_identity()}
    atomic_json(out/'validation.json',result);return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/mechanism_smoke')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    result = run(args.out, args.device)
    print(json.dumps({'status': result['status'], 'checks': result['checks'], 'wall_s': result['wall_s']}))

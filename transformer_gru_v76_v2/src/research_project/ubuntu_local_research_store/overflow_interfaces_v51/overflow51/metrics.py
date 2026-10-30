"""Risk first; abstentions never silently disappear from the denominator."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def risk_metrics(y, score, accepted=None):
    y=np.asarray(y,dtype=np.int64);score=np.asarray(score,dtype=float)
    accepted=np.ones(len(y),bool) if accepted is None else np.asarray(accepted,dtype=bool)
    if not len(y) or y.shape!=score.shape or y.shape!=accepted.shape or not np.isfinite(score).all():
        raise ValueError("nonempty aligned finite predictions required")
    if not np.isin(y,[0,1]).all() or ((score<0)|(score>1)).any():raise ValueError("invalid binary predictions")
    p=score>=.5;tp=int((p & accepted & (y==1)).sum());fp=int((p & accepted & (y==0)).sum())
    fn=int(((~p | ~accepted) & (y==1)).sum());tn=int((~p & accepted & (y==0)).sum())
    return {"n":len(y),"positive":int(y.sum()),"coverage":float(accepted.mean()),
            "refused":int((~accepted).sum()),"tp":tp,"fp":fp,"fn_including_refused_positive":fn,"tn":tn,
            "precision":tp/max(1,tp+fp),"recall_all_positive":tp/max(1,tp+fn),
            "f1_all_positive":2*tp/max(1,2*tp+fp+fn),"accuracy_refusals_wrong":(tp+tn)/len(y),
            "accepted_accuracy":float((p[accepted]==y[accepted]).mean()) if accepted.any() else None,
            "ap_all_scores":float(average_precision_score(y,score)) if y.sum() else None,
            "auroc_all_scores":float(roc_auc_score(y,score)) if len(np.unique(y))==2 else None,
            "brier_all_scores":float(np.mean((score-y)**2))}

def clustered_bootstrap(y,score,groups,accepted=None,repetitions=500,seed=51):
    y=np.asarray(y);score=np.asarray(score);groups=np.asarray(groups)
    accepted=np.ones(len(y),bool) if accepted is None else np.asarray(accepted)
    unique=np.unique(groups);indexes={g:np.flatnonzero(groups==g) for g in unique}
    if len(unique)<2:return {"status":"insufficient_independent_groups"}
    rng=np.random.default_rng(seed);values=[]
    for _ in range(repetitions):
        ix=np.concatenate([indexes[g] for g in rng.choice(unique,len(unique),replace=True)])
        values.append(risk_metrics(y[ix],score[ix],accepted[ix])["f1_all_positive"])
    return {"groups":len(unique),"repetitions":repetitions,"f1_percentile95":np.quantile(values,[.025,.975]).tolist(),
            "limitation":"conditional on supplied group definition, not producer-family independence"}

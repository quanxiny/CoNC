"""Class-prior-matched loss; the frozen continuation trainer is not edited."""
import math
from fractions import Fraction
from hybrid_retrain_core import supervised_ce as original_ce
ARM='class_prior_matched'


def constants(class_counts, residual_counts):
    if len(class_counts)!=2 or len(residual_counts)!=2:
        raise ValueError('binary full-TRAIN class counts required')
    for n,r in zip(class_counts,residual_counts):
        if type(n) is not int or type(r) is not int or n<=0 or not 0<=r<=n:
            raise ValueError('invalid full-TRAIN class/residual counts')
    c=[Fraction(n+r,n) for n,r in zip(class_counts,residual_counts)]
    mu=Fraction(sum(class_counts)+sum(residual_counts),sum(class_counts))
    return {'class_counts':list(class_counts),'residual_counts':list(residual_counts),
        'class_weight_numerators':[n+r for n,r in zip(class_counts,residual_counts)],
        'class_weight_denominators':list(class_counts),'class_weights':[float(x) for x in c],
        'class_total_weights':[n+r for n,r in zip(class_counts,residual_counts)],
        'mu':float(mu),'mu_numerator':mu.numerator,'mu_denominator':mu.denominator}


def classmatch_ce(logits,labels,R,arm,mu,class_weights):
    if arm!=ARM:raise ValueError('classmatch arm required')
    if len(class_weights)!=2 or any(type(c) not in (int,float) or not math.isfinite(c) or not 1<=c<=2 for c in class_weights):
        raise ValueError('two frozen full-TRAIN class weights required')
    # Preserve the original logits/label/R validation and per-row CE exactly.
    _,detail=original_ce(logits,labels,R,'ordinary_continue',mu)
    import torch
    w=torch.as_tensor(class_weights,dtype=detail['CE'].dtype,device=logits.device)[labels]
    return (w*detail['CE']).mean()/mu,{'CE':detail['CE'],'weights':w/mu,'R':detail['R']}

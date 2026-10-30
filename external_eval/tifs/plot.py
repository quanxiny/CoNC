"""Saved-CSV figures; portable serif font, no embedded title/caption."""
import argparse
from .common import *

def run(out):
    require_freeze(out)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from .analyze import csv_rows
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    dest=Path(read(out/'analysis_latest.json')['directory']);summary=csv_rows(dest/'summary.csv');pairs=csv_rows(dest/'pair_metrics.csv');figdir=dest/'figures';figdir.mkdir(exist_ok=True)
    if (figdir/'receipt.json').exists():
        prior=read(figdir/'receipt.json')
        if prior['source_analysis_sha256']!=sha(dest/'completion.json'):raise ValueError('Plot source changed')
        for name,h in prior['files'].items():
            if sha(figdir/name)!=h:raise ValueError('Completed plot changed')
        return prior
    fig,axes=plt.subplots(1,3,figsize=(10,3.6));colors={'full':'#173B60','mlp_q':'#A94E24','vd':'#4F7265','deepdfa':'#716078'}
    methods=config()['variants']+['vd','deepdfa']
    for ax,dataset in zip(axes,['source','mit','formai']):
        rr={r['method']:r for r in summary if r['dataset']==dataset and r['cohort']=='common_neural'};names=[m for m in methods if m in rr]
        values=[float(rr[m]['f1_mean']) if rr[m]['f1_mean'] else np.nan for m in names];sd=[float(rr[m]['f1_sd'] or 0) for m in names]
        ax.barh(range(len(names)),values,xerr=sd,color=[colors.get(m,'#647D96') for m in names],capsize=2,height=.7)
        ax.set_yticks(range(len(names)),[m.replace('_',' ') for m in names]);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel(dataset.upper()+' F1');ax.grid(axis='x',alpha=.15)
    fig.tight_layout();fig.savefig(figdir/'mechanism_f1.png',dpi=600,bbox_inches='tight');fig.savefig(figdir/'mechanism_f1.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7,3.2))
    for ax,dataset in zip(axes,['mit','diagnostic']):
        names=[];means=[];errors=[]
        for method in methods:
            values=[float(r['pair_accuracy']) for r in pairs if r['dataset']==dataset and r['method']==method and r['cohort']=='common_neural' and r['pair_accuracy']]
            if not values:continue
            names.append(method);means.append(np.mean(values));errors.append(np.std(values,ddof=1) if len(values)>1 else 0)
        ax.bar(range(len(names)),means,yerr=errors,color=[colors.get(m,'#647D96') for m in names],capsize=2)
        ax.set_xticks(range(len(names)),[m.replace('_',' ') for m in names],rotation=60,ha='right');ax.set_ylim(0,1);ax.set_ylabel(dataset.upper()+' pair accuracy');ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(figdir/'paired_boundary.png',dpi=600,bbox_inches='tight');fig.savefig(figdir/'paired_boundary.pdf',bbox_inches='tight');plt.close(fig)
    receipt={'status':'COMPLETE','source_analysis_sha256':sha(dest/'completion.json'),'font':'DejaVu Serif','dpi':600,'files':{p.name:sha(p) for p in figdir.iterdir() if p.suffix in ['.png','.pdf']},'captions':'Mechanism F1 and paired-boundary accuracy; error bars are sample SD across the three fixed seeds. Common neural IDs are used; pair denominators and coverage are in pair_metrics.csv.'}
    atomic_json(figdir/'receipt.json',receipt,immutable=True);return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=campaign_root());a=p.parse_args();print(json.dumps(run(output_dir(a.out))))

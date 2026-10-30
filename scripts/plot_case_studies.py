#!/usr/bin/env python3
"""Plot two saved-result case studies; no C execution or model/analyzer calls."""
from pathlib import Path
import argparse
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyBboxPatch
P=Path(__file__).resolve().parents[1];D=P/'data/case_studies'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=P/'outputs/case_figures')
args=parser.parse_args();O=args.output_dir.resolve();O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
source=json.loads((D/'source_87482.json').read_text())
mit=json.loads((D/'mit_basic_00001.json').read_text())
scenes={x['scene_id'].rsplit(':',1)[1]:x for x in mit['scenes']}
assert [int(source['predictions'][str(s)]['prediction']) for s in [11,29,47]]==[0,0,0]
assert int(source['eva']['prediction'])==1 and int(source['query']['label'])==0
assert all(int(scenes['min']['predictions'][str(s)]['prediction'])==0 for s in [11,29,47])
assert int(scenes['min']['eva_prediction']['prediction'])==1
assert max(float(scenes['min']['predictions'][str(s)]['risk']) for s in [11,29,47])<1.2e-8
ink='#173A54';teal='#17675B';rust='#A33E27';muted='#4C5861'
def text(ax,x,y,s,size=9,color=ink,weight='normal',**kw):
 return ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,va='top',**kw)
def box(ax,x,y,w,h,face='#F4F6F8',edge='#BCC7CE'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.007,rounding_size=0.012',linewidth=.65,edgecolor=edge,facecolor=face))
def panel(ax,kind):
 ax.set(xlim=(0,1),ylim=(.08,1));ax.axis('off')
 if kind=='source':
  text(ax,.02,.985,'(a) OverflowGen: safe loop',10,weight='bold')
  box(ax,.02,.61,.96,.30)
  code=['int a[322], i = 80, limit = 300;', 'int *p = a, *q = p;', 'while (i++ < limit) {', '    *(q + i) = 91;', '}']
  for j,line in enumerate(code):text(ax,.05,.885-j*.051,line,9.2)
  text(ax,.03,.575,'Writes use i = 81, ..., 300',9.5,teal,weight='bold')
  x0,x1,y=.05,.95,.455
  ax.add_patch(Rectangle((x0,y),x1-x0,.065,facecolor='white',edgecolor=ink,lw=.8))
  x=lambda v:x0+(x1-x0)*v/322
  ax.add_patch(Rectangle((x(81),y),x(301)-x(81),.065,facecolor='#C5E2DA',edgecolor=teal,lw=.8))
  for v,label in [(0,'0'),(81,'81'),(300,'300'),(322,'322')]:
   ax.plot([x(v),x(v)],[y-.01,y],color=ink,lw=.8)
   text(ax,x(v),y-.02,label,8,ha='right' if v==300 else 'center')
  text(ax,.03,.37,'Eva: slevel=100; iteration merging',8.7)
  text(ax,.03,.315,'Final i interval: [181, 2147483647]',8.7,rust)
  box(ax,.02,.10,.46,.135,face='#EFF7F3',edge=teal)
  box(ax,.52,.10,.46,.135,face='#FBF0EB',edge=rust)
  text(ax,.05,.213,'CoNC: safe (3/3)',9,teal,weight='bold')
  text(ax,.05,.157,'Correct classification',8.3,teal)
  text(ax,.55,.213,'Eva: possible OOB',9,rust,weight='bold')
  text(ax,.55,.157,'False alert',8.3,rust)
 else:
  text(ax,.02,.985,'(b) MIT: one-past-end write',10,weight='bold')
  box(ax,.02,.61,.96,.30)
  text(ax,.05,.885,'char buf[10];',9.2)
  text(ax,.05,.817,"buf[9] = 'A';",9.2,teal)
  text(ax,.53,.817,'safe version',8.7,teal)
  text(ax,.05,.749,"buf[10] = 'A';",9.2,rust)
  text(ax,.53,.749,'overflow version',8.7,rust)
  text(ax,.03,.575,'Capacity and access index both retained',9.3,weight='bold')
  x0,y,w=.06,.455,.071
  for i in range(11):
   fc='#F4F6F8' if i<9 else '#C5E2DA' if i==9 else '#F3CDC0'
   ec=ink if i<9 else teal if i==9 else rust
   ax.add_patch(Rectangle((x0+i*w,y),w,.065,facecolor=fc,edgecolor=ec,lw=.65))
   text(ax,x0+(i+.5)*w,y+.054,str(i),7.5,ec,ha='center')
  ax.annotate('',xy=(x0+10*w+.005,y+.093),xytext=(x0+10*w+.005,y-.035),arrowprops={'arrowstyle':'-','color':rust,'lw':1.3})
  text(ax,.03,.37,'ACCESS parents: index 10, capacity 10',8.7)
  text(ax,.03,.315,'CoNC: p < 1.2e-8; Eva: 10 < 10 is false',8.7,rust)
  box(ax,.02,.10,.46,.135,face='#FBF0EB',edge=rust)
  box(ax,.52,.10,.46,.135,face='#EFF7F3',edge=teal)
  text(ax,.05,.213,'CoNC: safe (3/3)',9,rust,weight='bold')
  text(ax,.05,.157,'Missed overflow',8.3,rust)
  text(ax,.55,.213,'Eva: certain alarm',9,teal,weight='bold')
  text(ax,.55,.157,'Correct detection',8.3,teal)
outputs=[]
fig,axes=plt.subplots(1,2,figsize=(6.5,2.50));fig.subplots_adjust(left=.005,right=.995,bottom=.005,top=.995,wspace=.07)
for ax,k in zip(axes,['source','mit']):panel(ax,k)
out=O/'fig6_case_comparison.png';fig.savefig(out,dpi=600);plt.close(fig);outputs.append(out)
for k in ['source','mit']:
 fig,ax=plt.subplots(figsize=(3.2,2.50));fig.subplots_adjust(left=.005,right=.995,bottom=.005,top=.995);panel(ax,k)
 out=O/f'case_{k}_diagnostic.png';fig.savefig(out,dpi=600);plt.close(fig);outputs.append(out)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'scope':'Retrospective case illustrations; not estimates of error prevalence.', 'source_id':source['unit_id'], 'mit_ids':[scenes[k]['scene_id'] for k in ['ok','min']], 'font':'matplotlib serif; installed fonts determine appearance', 'dpi':600, 'inputs':{p.name:sha(p) for p in [D/'source_87482.json',D/'mit_basic_00001.json']}, 'outputs':{p.name:sha(p) for p in outputs}}
(O/'case_figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))

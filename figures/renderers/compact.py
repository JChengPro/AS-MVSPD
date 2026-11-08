#!/usr/bin/env python3
"""Independent Bioinformatics figure package; existing manuscripts and results are read-only."""
import os
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(BASE/'source/mpl_cache')
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import sys,json,hashlib,shutil,runpy,textwrap,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.text import Text
from PIL import Image,ImageOps,ImageDraw
ROOT=BASE.parent
sys.path.insert(0,str(ROOT))
from stablejoint import analyze_multiview,generate_signs
from stablejoint.benchmark import evaluate_published_baselines,fast_stable_multiview_pvalue,evaluate_methods
ORIGINAL=ROOT/'figures/source_data'
HE=ROOT/'results/he_validation'
DATA=ROOT/'results/figure_inputs'; MAIN=BASE/'main'; SUPP=BASE/'supplementary'; DOC=BASE/'documentation'; REVIEW=BASE/'review'
for folder in [DATA,MAIN,SUPP,DOC,REVIEW]: folder.mkdir(parents=True,exist_ok=True)
COL={'Microbiome':'#23856B','Species':'#23856B','KO':'#7963A6','Metabolome':'#C47A32','Metabolite':'#C47A32','Joint IUT':'#333333'}
METHODS=['AS-MVSPD','Paired-aSPU-IUT','Paired-CQ-IUT','Paired-RankCOM-IUT']
MC=dict(zip(METHODS,['#426B94','#23856B','#C47A32','#7963A6']))
MARK=dict(zip(METHODS,['o','s','D','^']))
ORDER=['HADDAD Air','HADDAD IHH','MARS IBS 2020','HE infants 2–12 mo']
SHORT=['HADDAD\nAir','HADDAD\nIHH','MARS','HE infants']
MANIFEST=[]; SOURCES={}; CAPTIONS={}; QA=[]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def savejson(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write(name,frame): frame.to_csv(DATA/name,index=False,encoding='utf-8-sig')
def read(name): return pd.read_csv(DATA/name)
def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1,'lines.markersize':3.5,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white','axes.facecolor':'white','axes.titlepad':8})
style()
def decorate(ax,label,title):
    ax.set_title(title,loc='left',fontweight='bold',fontsize=8,pad=9)
    ax.text(-.13,1.055,label,transform=ax.transAxes,weight='bold',fontsize=10,va='bottom')
    ax.tick_params(length=2.5,width=.6)
    ax.grid(axis='y',color='#E8E8E8',lw=.45); ax.set_axisbelow(True)
def layout(r=2,c=2,h=145):
    fig,axes=plt.subplots(r,c,figsize=(178/25.4,h/25.4),squeeze=False)
    fig.subplots_adjust(left=.12,right=.965,bottom=.11,top=.91,wspace=.43,hspace=.43)
    return fig,axes

def emit(fig,stem,description=None,alt=None):
    stem=Path(stem); ident=stem.name
    # Final publication dimensions, fixed canvas; no bbox='tight' changing width.
    w,h=fig.get_size_inches()*25.4
    target=86 if w<110 else 178
    fig.set_size_inches(target/25.4,(h*target/w)/25.4,forward=True)
    if ident=='Figure_S5_local_complete':
        ax=fig.axes[1]; legend=ax.get_legend()
        if legend is not None:
            handles,labels=ax.get_legend_handles_labels()
            ax.legend(handles=handles,labels=labels,loc='lower left',bbox_to_anchor=(0,.12),fontsize=6)
        for t in list(ax.texts):
            if t.get_text().startswith('Decrease counts'): t.remove()
        ax=fig.axes[5]
        for t in list(ax.texts):
            if 'putative Betaine' in t.get_text(): t.remove()
        ax.text(.03,.95,'M118T213_pos\nPutative betaine\nRank 344; maxT P = 0.617',transform=ax.transAxes,va='top',fontsize=6,bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
    # Explanatory footer paragraphs belong in the separate figure legend file.
    for t in list(fig.texts):
        if t.get_position()[1]<.08 and len(t.get_text())>85: t.remove()
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():
            t.set_fontsize(max(6.5,t.get_fontsize()))
    for ax in fig.axes:
        for title in [ax.title,ax._left_title,ax._right_title]:
            txt=title.get_text()
            if txt and '\n' not in txt and len(txt)>30:
                title.set_text('\n'.join(textwrap.wrap(txt,30)))
                title.set_fontsize(min(title.get_fontsize(),8))
    def drawn_texts():
        ticktexts=set(); active=set()
        for ax in fig.axes:
            for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
                low,high=sorted(limits)
                for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                    ticktexts.update([tick.label1,tick.label2])
                    if low-1e-10<=tick.get_loc()<=high+1e-10: active.update([tick.label1,tick.label2])
        return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and (t not in ticktexts or t in active)]
    for attempt in range(3):
        fig.canvas.draw(); renderer=fig.canvas.get_renderer(); fw,fh=fig.canvas.get_width_height()
        boxes=[t.get_window_extent(renderer) for t in drawn_texts()]
        x0=min([0]+[b.x0 for b in boxes]); x1=max([fw]+[b.x1 for b in boxes]); y0=min([0]+[b.y0 for b in boxes]); y1=max([fh]+[b.y1 for b in boxes])
        if x0>=-1 and y0>=-1 and x1<=fw+1 and y1<=fh+1: break
        left=max(0,-x0/fw+.012); right=max(0,(x1-fw)/fw+.012)
        bottom=max(0,-y0/fh+.012); top=max(0,(y1-fh)/fh+.012)
        for ax in fig.axes:
            pos=ax.get_position(); ax.set_position([left+pos.x0*(1-left-right),bottom+pos.y0*(1-top-bottom),pos.width*(1-left-right),pos.height*(1-top-bottom)])
        for t in fig.texts:
            x,y=t.get_position(); t.set_position((left+x*(1-left-right),bottom+y*(1-top-bottom)))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer(); fw,fh=fig.canvas.get_width_height()
    outside=[]
    for t in drawn_texts():
        bb=t.get_window_extent(renderer)
        if bb.width and bb.height and (bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fw+1 or bb.y1 > fh+1):
            outside.append({'text':t.get_text()[:100],'bbox':[round(v,1) for v in bb.bounds]})
    QA.append({'figure':ident,'out_of_canvas_text':outside})
    formats=globals().get('EXPORT_FORMATS',['pdf','svg','tif','png'])
    for ext in ['pdf','svg']:
        if ext in formats:fig.savefig(stem.with_suffix('.'+ext),facecolor='white')
    for ext,dpi in [('tif',1200),('png',300)]:
        if ext not in formats:continue
        kw={'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tif' else {}
        fig.savefig(stem.with_suffix('.'+ext),dpi=dpi,facecolor='white',**kw)
        with Image.open(stem.with_suffix('.'+ext)) as raster:rgb=raster.convert('RGB')
        if ext=='tif':rgb.save(stem.with_suffix('.tif'),compression='tiff_lzw',dpi=(1200,1200))
        else:rgb.save(stem.with_suffix('.png'),dpi=(300,300))
        rgb.close()
    width,height=fig.get_size_inches()*25.4
    MANIFEST.append({'figure':ident,'width_mm':round(width,4),'height_mm':round(height,4),
                     'tiff_dpi':1200 if 'tif' in formats else None,'review_png_dpi':300 if 'png' in formats else None,
                     'tiff_color_mode':'RGB' if 'tif' in formats else None,'formats':formats})
    if description: CAPTIONS[ident]={'legend':description,'alt_text':alt or description}
    print('Rendered',ident,flush=True)




def main1():
    fig=plt.figure(figsize=(178/25.4,88/25.4)); gs=fig.add_gridspec(1,3,width_ratios=[1,1.55,1],left=.055,right=.98,bottom=.07,top=.90,wspace=.22)
    aa=[fig.add_subplot(gs[i]) for i in range(3)]
    for ax,l,t in zip(aa,'ABC',['Paired multi-omics','Shared calibration and inference','Bounded feature evidence']):
        decorate(ax,l,t); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax=aa[0]
    for j,(label,col) in enumerate([('Microbiome',COL['Microbiome']),('Function',COL['KO']),('Metabolome',COL['Metabolome'])]):
        yy=.80-j*.26; ax.text(.5,yy+.08,label,color=col,ha='center',weight='bold')
        for xx,txt in [(.10,'Pre'),(.61,'Post')]:
            ax.add_patch(FancyBboxPatch((xx,yy-.08),.29,.13,boxstyle='round,pad=.015',fc='#F2F5F7',ec=col,lw=.8)); ax.text(xx+.145,yy-.015,txt,ha='center',va='center')
        ax.annotate('',xy=(.59,yy-.015),xytext=(.41,yy-.015),arrowprops={'arrowstyle':'->','lw':.8})
    ax.text(.5,.02,'Same individuals across views\nDifference = Post − Pre',ha='center',va='bottom')
    ax=aa[1]
    texts=['Aligned paired differences','Shared individual sign flips','Feature scores → top-k aggregation','Recalibrate adaptive evidence']
    for i,txt in enumerate(texts):
        yy=.88-i*.19; ax.add_patch(FancyBboxPatch((.04,yy-.10),.92,.13,boxstyle='round,pad=.015',fc='#F1F5F8',ec='#426B94',lw=.7)); ax.text(.50,yy-.035,txt,ha='center',va='center',fontsize=6.8)
        if i<3: ax.annotate('',xy=(.5,yy-.16),xytext=(.5,yy-.105),arrowprops={'arrowstyle':'->','lw':.8})
    ax.text(.25,.08,'View P-values\n↓\nJoint P = max(view P)',ha='center',va='center',fontsize=6.5)
    ax.text(.79,.08,'Separate feature tests\n↓\nWithin-view maxT',ha='center',va='center',fontsize=6.5)
    ax=aa[2]
    vals=[np.ones(8),np.array([8,0,0,0,0,0,0,0]),np.array([1,-1,1,-1,1,-1,1,-1])]
    for i,(txt,v) in enumerate(zip(['Concordant','One individual','Opposing'],vals)):
        yy=.79-i*.27; score=abs(v.sum())/np.sqrt((v*v).sum())
        ax.text(0,yy+.09,txt,weight='bold'); ax.text(1,yy+.09,f'S = {score:.2f}',ha='right')
        for j,value in enumerate(v): ax.plot([.05+j*.125]*2,[yy-.05,yy-.05+value*.008],color='#426B94' if value>=0 else '#B66B73',lw=3)
        ax.axhline(yy-.05,color='#DDDDDD',lw=.6)
    ax.text(.5,.015,r'$S=|\sum_i d_i|/\sqrt{\sum_i d_i^2}\leq\sqrt{n}$',ha='center')
    emit(fig,MAIN/'Figure_1_method_framework','AS-MVSPD workflow. (A) Alignment of paired measurements across views. (B) Shared individual-level sign calibration, adaptive aggregation, joint intersection–union testing and separate within-view maxT analysis. (C) Illustrative self-normalized scores; examples explain the statistic and are not experimental results.','Three panels show paired input alignment, the inference workflow, and bounded scores for concordant, isolated and opposing changes.'); plt.close(fig)


def main3():
    ov=read('public_dataset_overview.csv').set_index('display_label').loc[ORDER]
    fig,axes=layout(h=132); a,b,c,d=axes.flat
    decorate(a,'A','Paired individuals and retained features')
    for i,(ds,r) in enumerate(ov.iterrows()):
        yy=3-i; a.plot([r.microbiome_features,r.metabolome_features],[yy]*2,color='#D4DCE1',lw=1.2)
        for key,marker,col in [('microbiome_features','o',COL['Microbiome']),('metabolome_features','s',COL['Metabolome'])]: a.scatter(r[key],yy,c=col,marker=marker,s=20)
        a.text(1.02,yy,f'n={int(r.paired_subjects)}',transform=a.get_yaxis_transform(),va='center',fontsize=6.5)
    a.set_yticks(range(4),['HE infants','MARS','HADDAD IHH','HADDAD Air']); a.set_xscale('log'); a.set_xlabel('Features per view (log scale)'); a.set_ylim(-.5,3.5); a.grid(False)
    a.legend(handles=[Line2D([],[],ls='',marker=mk,color=COL[k],label=k) for k,mk in [('Microbiome','o'),('Metabolome','s')]],loc='lower right',fontsize=6)
    for ax,label,title in [(b,'B','View-specific and joint evidence'),(c,'C','Comparison of paired testing statistics')]: decorate(ax,label,title); ax.axhline(-np.log10(.05),color='#777',ls='--',lw=.7); ax.set_ylim(0,2.9); ax.set_xticks(range(4),SHORT); ax.set_ylabel(r'$-\log_{10}(P)$')
    handles=[]
    for offset,(key,lab,mk,col) in zip([-.22,0,.22],[('microbiome_p','Microbiome','o',COL['Microbiome']),('metabolome_p','Metabolome','s',COL['Metabolome']),('joint_p','Joint IUT','D','#333')]):
        b.scatter(np.arange(4)+offset,-np.log10(ov[key]),s=19,marker=mk,color=col); handles.append(Line2D([],[],ls='',marker=mk,color=col,label=lab))
    b.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.03),ncol=3,fontsize=6,handletextpad=.3,columnspacing=.6)
    md=read('real_data_published_baselines.csv')
    for off,method in zip(np.linspace(-.25,.25,4),METHODS):
        r=md.loc[md.data_scope.eq('public')&md.method.eq(method)].set_index('dataset').loc[ORDER]
        c.scatter(np.arange(4)+off,-np.log10(r.joint_p),marker=MARK[method],color=MC[method],s=22,facecolors=MC[method] if method=='AS-MVSPD' else 'none',linewidths=.8)
    c.legend(handles=[Line2D([],[],marker=MARK[m],ls='',color=MC[m],markerfacecolor=MC[m] if m=='AS-MVSPD' else 'white',label=m.replace('Paired-','').replace('-IUT','')) for m in METHODS],loc='lower left',ncol=2,fontsize=6.5)
    decorate(d,'D','Joint significance under perturbation'); d.set_ylabel('Fraction with joint P ≤ 0.05'); d.set_xticks(range(4),SHORT); d.set_ylim(-.05,1.16)
    loo=read('public_leave_one_subject.csv'); sub=read('public_feature_resampling.csv'); sub=sub[sub.result_level.eq('Joint IUT')]
    for i,ds in enumerate(ORDER):
        for off,frame,key,mk in [(-.12,loo,'joint_p','o'),(.12,sub,'pvalue','s')]:
            pp=frame.loc[frame.display_label.eq(ds),key]; n=int((pp<=.05).sum()); fraction=n/len(pp)
            d.scatter(i+off,fraction,s=23,marker=mk,facecolors='white',edgecolors='#333')
            d.annotate(f'{n}/{len(pp)}',(i+off,fraction),xytext=(0,-13 if off<0 else 8),textcoords='offset points',ha='center',fontsize=6)
    d.legend(handles=[Line2D([],[],marker=m,ls='',mfc='white',color='#333',label=l) for m,l in [('o','Individual deletion'),('s','80% feature subsets')]],loc='lower left',fontsize=6.5)
    emit(fig,MAIN/'Figure_3_public_validation','Public paired microbiome–metabolome applications. (A) Paired sample sizes and retained features. (B) View and joint P-values. (C) AS-MVSPD and paired aSPU-, Chen–Qin- and RankCOM-type statistics under the same sign group and intersection–union rule. (D) Significant joint tests after individual deletion and 30 independent 80% feature subsets. HADDAD uses all 256 patterns; MARS and HE use closed 256-element subgroups. HE pairs 2- and 12-month samples from 57 infants. Study-specific P-values are not adjusted across datasets.','Four panels compare sample dimensions, paired test evidence, four methods and perturbation stability in HADDAD Air, HADDAD IHH, MARS and HE infants.'); plt.close(fig)


def s3():
    # Retain unchanged simulation and runtime panels from their validated source renderer.
    ns=load_renderer()
    oldemit=ns['save_publication_figure']
    def capture(fig,stem,**kw):
        ax=fig.axes[3]; ax.clear(); decorate(ax,'D','Closed-subgroup seed sensitivity')
        seed=read('public_seed_sensitivity.csv')
        for i,ds in enumerate(ORDER):
            vals=seed.loc[seed.display_label.eq(ds),'joint_p'].to_numpy(); ax.scatter(i+np.linspace(-.10,.10,len(vals)),vals,s=10,color='#426B94'); ax.hlines(np.median(vals),i-.15,i+.15,color='#333',lw=1)
        ax.axhline(.05,c='#777',ls='--',lw=.7); ax.set_yscale('log'); ax.set_ylim(.0025,.12); ax.set_xticks(range(4),['Air','IHH','MARS','HE']); ax.set_ylabel('Joint P (log scale)')
        ax=fig.axes[4]; ax.clear(); decorate(ax,'E','Adaptive aggregation scale')
        sub=read('public_feature_resampling.csv'); sub=sub[sub.result_level.isin(['Microbiome','Metabolome'])]
        colors=['#426B94','#7492B0','#9DB2C6','#C4D1DE','#E0E7EF']; bottom=np.zeros(4)
        for k,col in zip([1,2,5,10,20],colors):
            vals=np.array([sub.loc[sub.display_label.eq(ds),'selected_k'].eq(k).mean() for ds in ORDER]); ax.bar(range(4),vals,bottom=bottom,width=.65,color=col,edgecolor='white',lw=.4,label=f'k={k}'); bottom+=vals
        ax.set_xticks(range(4),['Air','IHH','MARS','HE']); ax.set_ylim(0,1.22); ax.set_yticks([0,.5,1]); ax.set_ylabel('Selection frequency'); ax.legend(ncol=3,loc='upper center',fontsize=6)
        fig.set_size_inches(178/25.4,138/25.4)
        fig.subplots_adjust(left=.08,right=.98,top=.90,bottom=.12,wspace=.48,hspace=.43)
        fig.axes[5].set_title('Runtime (three views)',loc='left',fontweight='bold',fontsize=8)
        first=fig.axes[0]; first.legend(handles=[Line2D([],[],color=ns['METHOD_COLORS'][m],marker=ns['METHOD_MARKERS'][m],label=m,mfc='white',lw=.8) for m in ns['METHOD_ORDER']],loc='upper left',ncol=1,fontsize=6)
        oldemit(fig,stem)
    ns['save_publication_figure']=capture; ns['make_supplementary3']()


def s4():
    fig,axes=layout(r=1,c=3,h=90); fig.subplots_adjust(left=.08,right=.98,top=.84,bottom=.23,wspace=.52)
    frames=[read('public_leave_one_subject.csv'),read('public_feature_resampling.csv'),read('public_seed_sensitivity.csv')]
    frames[1]=frames[1][frames[1].result_level.eq('Joint IUT')]
    for ax,l,title,df,key in zip(axes.flat,'ABC',['Individual deletion','80% feature subsets','Subgroup seed sensitivity'],frames,['joint_p','pvalue','joint_p']):
        decorate(ax,l,title)
        for i,ds in enumerate(ORDER):
            vals=df.loc[df.display_label.eq(ds),key].to_numpy(); ax.scatter(i+np.linspace(-.11,.11,len(vals)),vals,s=8,color='#426B94',alpha=.7); ax.hlines(np.median(vals),i-.19,i+.19,color='#333',lw=1)
        ax.axhline(.05,c='#777',ls='--',lw=.7); ax.set_yscale('log'); ax.set_ylim(.0025,.12); ax.set_xticks(range(4),['Air','IHH','MARS','HE'],rotation=30); ax.set_ylabel('Joint P (log scale)')
    emit(fig,SUPP/'Figure_S4_public_sensitivity','Public-data sensitivity analyses. (A) Individual deletion, with analysis-ready features locked. (B) Thirty 80% feature subsets. (C) Twenty closed-subgroup seeds for MARS and HE; HADDAD full-group P-values are reference points. Horizontal dashed lines indicate P = 0.05. All individual results are in the accompanying source tables.','Three panels show P-values after individual deletion, feature subsampling and subgroup selection in four public subsets.'); plt.close(fig)


def main4():
    fig=plt.figure(figsize=(178/25.4,185/25.4))
    gs=fig.add_gridspec(3,2,left=.13,right=.93,bottom=.07,top=.94,hspace=.36,wspace=.67,height_ratios=[.65,1.30,1.25])
    aa=[fig.add_subplot(gs[i,j]) for i in range(3) for j in range(2)]
    blocks=read('local_block_summary.csv'); reps=read('local_representative_features.csv'); heat=read('local_representative_heatmap.csv'); pre=read('local_representative_prepost.csv'); counts=read('local_fwer_direction_counts.csv')
    for ax,l,t in zip(aa,'ABCDEF',['Paired rat study','View-specific and joint tests','Representative feature scores','Individual paired changes','Selected paired measurements','Direction of maxT discoveries']): decorate(ax,l,t)
    ax=aa[0]; ax.axis('off')
    ax.text(.5,.85,'8 rats: before and after hyperbaric exposure',ha='center',fontsize=7)
    for i,col in enumerate(['#426B94','#B66B73']):
        ax.scatter(np.arange(8),np.full(8,.58-i*.28),c=col,s=16)
    for i in range(8): ax.plot([i,i],[.30,.58],c='#CCD3DA',lw=.7,zorder=0)
    ax.set_xlim(-1,8); ax.set_ylim(-.15,1.15); ax.text(-.35,.58,'Pre',va='center',ha='right',fontsize=6.5); ax.text(-.35,.30,'Post',va='center',ha='right',fontsize=6.5)
    ax.text(3.5,-.03,'745 Species · 5,475 KO · 1,029 metabolites',ha='center',fontsize=6.5)
    ax=aa[1]; labs=['Species','KO','Metabolite','Joint IUT']; pv=blocks.block_exact_p.tolist()+[max(blocks.block_exact_p)]
    ax.scatter(-np.log10(pv),np.arange(4),c=[COL[k] for k in labs],s=25); ax.set_yticks(range(4),labs); ax.invert_yaxis(); ax.set_xlim(0,2.7); ax.axvline(-np.log10(.05),c='#777',ls='--',lw=.7); ax.set_xlabel(r'$-\log_{10}(P)$')
    for i,p in enumerate(pv): ax.text(-np.log10(p)+.08,i,f'{p:.4f}',va='center',fontsize=6)
    # Short metabolite IDs keep raw identity traceable; full names stay in source table.
    names=[]
    for r in reps.itertuples():
        label=r.display_label
        if r.block=='Metabolite': label=str(r.display_label)
        if r.block=='KO': label=str(r.feature)
        
        if r.block=='Species':
            words=label.split(); label=words[0][0]+'. '+' '.join(words[1:])
        names.append(label if r.block=='Metabolite' else label.replace('_',' '))
    ax=aa[2]; yy=np.arange(len(reps)); ax.barh(yy,reps.signed_stable_score,color=[COL[k] for k in reps.block],height=.65); ax.set_yticks(yy,names,fontsize=6); ax.invert_yaxis(); ax.axvline(0,c='#777',lw=.6); ax.set_xlabel('Signed self-normalized score'); ax.set_xlim(-3.2,3.2)
    # Place long taxon names inside a generous dedicated left margin.
    pos=ax.get_position(); ax.set_position([pos.x0+.025,pos.y0,pos.width-.025,pos.height])
    ax=aa[3]; mat=heat.pivot(index='feature',columns='animal_id',values='sd_scaled_change').reindex(reps.feature)
    bound=max(abs(mat.to_numpy()).max(),1); im=ax.imshow(mat,aspect='auto',cmap='RdBu_r',vmin=-bound,vmax=bound)
    ax.set_yticks(range(len(reps)),[str(i+1) for i in range(len(reps))]); ax.set_xticks(range(8),range(1,9)); ax.set_xlabel('Rat'); ax.set_ylabel('Feature row (same order as C)'); ax.grid(False)
    cb=fig.colorbar(im,ax=ax,fraction=.06,pad=.04); cb.set_label('Paired change / SD',fontsize=6.5)
    ax=aa[4]; ax.set_axis_off(); pos=ax.get_position()
    examples=[pre.feature.unique()[0],'K02116','K01692',pre.loc[pre.block.eq('Metabolite'),'feature'].iloc[0]]
    for i,f in enumerate(examples):
        col=i%2; row=i//2
        inner=fig.add_axes([pos.x0+col*pos.width*.61,pos.y0+pos.height*(.08+(.48 if row==0 else 0)),pos.width*.39,pos.height*.29])
        p=pre[pre.feature.eq(f)]; pivot=p.pivot(index='animal_id',columns='time',values='value')
        for _,record in pivot.iterrows(): inner.plot([0,1],record[['Pre','Post']],c='#AAB5BF',lw=.65,zorder=0)
        inner.scatter(np.zeros(len(pivot)),pivot.Pre,c='#426B94',s=9); inner.scatter(np.ones(len(pivot)),pivot.Post,c='#B66B73',s=9)
        name='S. xylosus' if i==0 else str(f) if i in [1,2] else 'M109T115_pos'
        inner.set_title(name,fontsize=6.5,pad=3); inner.set_xticks([0,1],['Pre','Post'])
        scale='CLR' if i==0 else 'log2(ppm + 0.5)' if i in [1,2] else 'log2 intensity'
        inner.set_ylabel(scale,fontsize=6); inner.tick_params(labelsize=6,pad=1)
    ax=aa[5]
    for i,r in enumerate(counts.itertuples()):
        ax.barh(i,r.up/r.total,color=COL[r.block],height=.55); ax.barh(i,r.down/r.total,left=r.up/r.total,color=COL[r.block],alpha=.3,height=.55)
        ax.text(1.03,i,f'{r.up} ↑ / {r.down} ↓',va='center',fontsize=6.5)
    ax.set_yticks(range(3),counts.block); ax.set_xlim(0,1.45); ax.set_xticks([0,.5,1],['0','0.5','1']); ax.set_xlabel('Fraction of significant features')
    emit(fig,MAIN/'Figure_4_local_application','Paired rat application. (A) Eight matched rats before and after exposure. (B) View-specific and joint exact sign-flip tests. (C) Four species, three KO and six representative metabolite features passing within-view maxT adjustment. Metabolite feature IDs are used; full names and annotations are in source tables. (D) Paired changes divided by feature-specific SD across rats; rows follow C. (E) Four representative trajectories on their original analysis scales; metabolite intensity is sum-normalized before log2(x+1). (F) Increasing (dark) and decreasing (light) maxT discoveries. No causal or cross-view interaction claim follows from joint significance.','Six panels show the rat study, joint evidence, representative features, rat-level changes, paired measurements and discovery directions.'); plt.close(fig)


def he_supplement():
    fig,axes=layout(h=125); a,b,c,d=axes.flat
    decorate(a,'A','Complete-pair retention')
    retention=pd.read_csv(HE/'further_audit/retention_by_feeding_group.csv'); labels=['Breast milk','Experimental','Standard']
    a.barh(range(3),retention.all_subjects,color='#DEE5EB',height=.65); a.barh(range(3),retention.paired_subjects,color='#426B94',height=.65); a.set_yticks(range(3),labels); a.set_xlabel('Infants'); a.set_xlim(0,35)
    for i,r in enumerate(retention.itertuples()): a.text(r.all_subjects+1,i,f'{r.paired_subjects}/{r.all_subjects}',va='center',fontsize=6.5)
    decorate(b,'B','Two visits and retained views'); b.axis('off')
    for i,txt in enumerate(['80 infants · 277 available samples','2 months → 12 months','57 infants · 114 matched samples','33 microbial genera · 119 metabolites']): b.text(.5,.90-i*.25,txt,ha='center',va='center',fontsize=7)
    decorate(c,'C','Within-feeding-group longitudinal tests')
    summary=json.loads((HE/'further_audit/summary.json').read_text()); strata=[r for r in summary['other_sensitivities'] if r['analysis'].startswith('feeding_stratum')]
    for off,key,col in [(-.12,'microbiome_p',COL['Microbiome']),(.12,'metabolome_p',COL['Metabolome'])]: c.scatter(np.arange(3)+off,[-np.log10(r[key]) for r in strata],c=col,s=20)
    c.set_xticks(range(3),['Breast milk\nn=13','Experimental\nn=21','Standard\nn=23']); c.set_ylim(0,2.9); c.axhline(-np.log10(.05),ls='--',c='#777',lw=.7); c.set_ylabel(r'$-\log_{10}(P)$')
    c.legend(handles=[Line2D([],[],marker='o',ls='',color=COL[k],label=k) for k in ['Microbiome','Metabolome']],loc='lower left',fontsize=6.5)
    decorate(d,'D','Transformations and filtering')
    wanted=['primary_reproduction','metabolite_raw_concentration','metabolite_asinh_concentration','metabolite_log_half_min_positive_per_feature','metabolite_detected_25pct','microbiome_prevalence_0.1','microbiome_prevalence_0.5']
    rows=[next(r for r in summary['other_sensitivities'] if r['analysis']==k) for k in wanted]
    labels=['Primary','Raw metabolites','asinh metabolites','Log + half-minimum','Metabolite detection ≥25%','Microbe detection ≥10%','Microbe detection ≥50%']
    d.scatter([-np.log10(r['joint_p']) for r in rows],range(len(rows)),c='#426B94',s=17); d.set_yticks(range(len(rows)),labels,fontsize=6); d.invert_yaxis(); d.set_xlim(0,2.9); d.axvline(-np.log10(.05),c='#777',ls='--',lw=.7); d.set_xlabel(r'$-\log_{10}(P_{joint})$')
    pos=d.get_position(); d.set_position([pos.x0+.08,pos.y0,pos.width-.08,pos.height])
    emit(fig,SUPP/'Figure_S10_HE_quality_and_robustness','HE infant application and checks. (A) Complete-pair retention within original feeding groups. (B) Fixed comparison of 2 versus 12 months. (C) Longitudinal tests within each original feeding stratum, with preprocessing repeated within stratum. These are developmental/dietary changes, not between-arm treatment effects. (D) Selected prespecified audit transformations and detection thresholds; all audit results are supplied as tables. P-values use 256-element closed subgroups and are not multiplicity-adjusted across sensitivity analyses. The 4096-pattern audit gave joint P = 0.000244140625. Zero values and incomplete follow-up remain limitations; sign invariance is not established by these checks.','Four panels show infant retention, paired visits, feeding-stratum results and sensitivity to transformations and filtering.'); plt.close(fig)











"""Render main Figures 1-4 and Supplementary Figures S1-S10 from numeric tables."""
import argparse,json,runpy,shutil,sys,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope',choices=['all','public'],default='public',help='all requires the rat figure-source tables')
    p.add_argument('--source-data',type=Path,help='Use fully recomputed figure tables instead of the archived tables')
    p.add_argument('--rat-source',type=Path,help='Directory with local_* and qc_* rat tables')
    p.add_argument('--formats',nargs='+',choices=['pdf','svg','png','tif'],default=['pdf','svg','png','tif'])
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('as_mvspd_compact',ROOT/'figures/renderers/compact.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);ns=module.__dict__
    data=ns['DATA'];data.mkdir(parents=True,exist_ok=True)
    for f in data.glob('*'):
        if f.is_file():f.unlink()
    for f in (a.source_data or ROOT/'figures/source_data').glob('*'):
        if f.is_file():shutil.copy2(f,data/f.name)
    if a.scope=='all':
        rat=a.rat_source or ROOT/'.private/figure_source'
        if not rat.is_dir():raise FileNotFoundError('Supply rat figure tables using --rat-source; see docs/DATA.md')
        for f in rat.glob('*'):
            if f.is_file():shutil.copy2(f,data/f.name)
    audit=ROOT/'results/he_validation/further_audit';audit.mkdir(parents=True,exist_ok=True)
    # Rendering archived results never silently recomputes tests.
    for src,target in [('he_summary.json','summary.json'),('he_retention_by_feeding_group.csv','retention_by_feeding_group.csv')]:
        if not (audit/target).exists():shutil.copy2(ROOT/'reference'/src,audit/target)
    emit=ns['emit']
    ns['EXPORT_FORMATS']=a.formats
    emit_selected=emit
    def load_renderer():
        spec=importlib.util.spec_from_file_location('as_mvspd_panels',ROOT/'figures/renderers/legacy_panels.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);r=module.__dict__
        r.update(SOURCE=data,OUT_MAIN=ns['MAIN'],OUT_SUPP=ns['SUPP'],DATASET_ORDER=ns['ORDER'],save_publication_figure=emit_selected)
        r['OMIC_COLORS'].update({k:v for k,v in ns['COL'].items() if k in ['Species','KO','Metabolite','Joint IUT']})
        r['METHOD_COLORS'].update(ns['MC']);r['METHOD_COLORS'].update({'AS-MVSPD+LOO':'#94A9BE','Max-MV':'#23856B','Dense-MV':'#777777'})
        r['TIME_COLORS'].update({'Pre':'#426B94','Post':'#B66B73'});r['apply_publication_style']=ns['style'];r['apply_dense_main_style']=ns['style']
        # runpy returns a dictionary distinct from function globals: patch each function namespace.
        for v in list(r.values()):
            if callable(v) and getattr(v,'__module__',None)=='<run_path>':v.__globals__.update(r)
        ns['style']();return r
    ns['s3'].__globals__['load_renderer']=load_renderer
    ns['main1']();r=load_renderer();r['make_figure2']();ns['main3']()
    if a.scope=='all':ns['main4']();r['make_supplementary1']()
    r['make_supplementary2']();ns['s3']();ns['s4']()
    if a.scope=='all':r['make_supplementary5']()
    for fn in ['make_supplementary6','make_supplementary7','make_supplementary8']:ns['style']();r[fn]()
    fixed=runpy.run_path(str(ROOT/'figures/renderers/fixed_scale.py'),init_globals={'DATA':data,'OUT_SUPP':ns['SUPP'],'emit':emit_selected})
    fixed['render_supplementary9']();ns['style']();ns['he_supplement']()
    doc=ns['DOC'];doc.mkdir(parents=True,exist_ok=True)
    legends=json.loads((ROOT/'figures/legends.json').read_text())
    lines=['# Manuscript figure legends','']
    for key,text in legends.items():
        if a.scope=='public' and key in ['4','S1','S5']:continue
        lines.extend(['## Figure '+key,'',text,''])
    (doc/'LEGENDS.md').write_text('\n'.join(lines))
    (doc/'manifest.json').write_text(json.dumps(ns['MANIFEST'],indent=2)+'\n')
    (doc/'layout_preflight.json').write_text(json.dumps(ns['QA'],indent=2)+'\n')
    print('Rendered',len(ns['MANIFEST']),'figures; source tables unchanged.')
if __name__=='__main__':main()

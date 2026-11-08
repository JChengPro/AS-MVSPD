"""Prepare the three public studies, or user-supplied rat omics inputs."""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from stablejoint.preprocessing import load_haddad,load_mars,load_he,sha256


def public_inputs():
    raw=ROOT/'data/raw'
    for ds,loader in [('HADDAD_OSA_IHH',load_haddad),('MARS_IBS_2020',load_mars),('HE_INFANTS_MFGM_2019',load_he)]:
        x,y,pairs,audit=loader(raw)
        folder=ROOT/'data/public_prepared'/ds;folder.mkdir(parents=True,exist_ok=True)
        x.to_csv(folder/'microbiome_paired_changes.csv');y.to_csv(folder/'metabolome_paired_changes.csv')
        pairs.to_csv(folder/'pair_mapping.csv',index=False)
        (folder/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        print(ds,len(x),x.shape[1],y.shape[1])


def local_inputs(config):
    from stablejoint import local_preprocessing as core
    from stablejoint.local_inputs import configured_paths
    paths=configured_paths(ROOT,config)
    mapping=core.load_sample_map(paths.sample_map)
    species,species_qc,species_summary=core.load_species(paths,mapping)
    metabolite,metabolite_qc,metabolite_meta,normalization=core.load_metabolites(paths,mapping)
    _,ko,ko_stats=core.load_ko_delta(paths,mapping,core)
    if not species.index.equals(ko.index) or not species.index.equals(metabolite.index):raise ValueError('Unaligned rat identifiers')
    folder=ROOT/'data/local';folder.mkdir(parents=True,exist_ok=True)
    frames={'species_paired_changes':species,'ko_paired_changes':ko,'metabolite_paired_changes':metabolite,
            'species_feature_qc':species_qc,'species_abundance_summary':species_summary,'ko_univariate_statistics':ko_stats,
            'metabolite_feature_qc':metabolite_qc,'metabolite_annotation_metadata':metabolite_meta,
            'metabolite_normalization_audit':normalization,'sample_mapping':mapping}
    for name,df in frames.items():df.to_csv(folder/(name+'.csv'),index=name.endswith('paired_changes'))
    configfile=config or Path(__import__('os').environ.get('AS_MVSPD_LOCAL_INPUTS',ROOT/'.private/local_input_paths.json'))
    rawpaths=json.loads(configfile.read_text())
    annotation=Path(rawpaths['ko_annotation']).expanduser()
    if not annotation.is_absolute():annotation=configfile.parent/annotation
    fields=['KOEntry','GeneName','KODescription','EC','PathwayDefinition','KEGGLevel1','KEGGLevel2']
    a=pd.read_csv(annotation,sep='\t',usecols=fields,low_memory=False).dropna(subset=['KOEntry'])
    a=a.groupby('KOEntry',as_index=False).agg(**{new:(old,lambda v:';'.join(sorted(set(v.dropna().astype(str))))) for new,old in
        [('gene_name','GeneName'),('ko_description','KODescription'),('ec','EC'),('pathway_definition','PathwayDefinition'),('kegg_level1','KEGGLevel1'),('kegg_level2','KEGGLevel2')]})
    a.to_csv(folder/'ko_annotation_metadata.csv',index=False)
    sourcepaths={'sample_mapping':str(paths.sample_map),'species':str(paths.species),'ko':str(paths.ko),'metabolite':str(paths.metabolite),'ko_annotation':str(annotation)}
    audit={'subjects':len(mapping),'blocks':{'Species':list(species.shape),'KO':list(ko.shape),'Metabolite':list(metabolite.shape)},
           'source_paths':sourcepaths,'source_sha256':{k:sha256(Path(v)) for k,v in sourcepaths.items()}}
    (folder/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('rats',audit['blocks'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope',choices=['public','local','all'],default='public')
    p.add_argument('--inputs',type=Path,help='Rat paths JSON, relative paths resolve against that JSON')
    a=p.parse_args()
    if a.scope in ['public','all']:public_inputs()
    if a.scope in ['local','all']:local_inputs(a.inputs)
if __name__=='__main__':main()

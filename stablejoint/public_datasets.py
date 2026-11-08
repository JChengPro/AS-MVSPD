"""Explicit current-manuscript datasets and aligned input validation."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_pair(folder):
    x=pd.read_csv(folder/'microbiome_paired_changes.csv',index_col=0)
    y=pd.read_csv(folder/'metabolome_paired_changes.csv',index_col=0)
    pairs=pd.read_csv(folder/'pair_mapping.csv',dtype={'subject_id':str},keep_default_na=False)
    for frame in [x,y]:
        frame.index=frame.index.astype(str)
        if not frame.index.is_unique or not frame.columns.is_unique:raise ValueError('Duplicate identifiers')
        if not np.isfinite(frame.to_numpy(dtype=float)).all():raise ValueError('Non-finite matrix')
    if not x.index.equals(y.index):raise ValueError('Omics subject order differs')
    if not x.index.tolist()==pairs.subject_id.astype(str).tolist():raise ValueError('Pair map order differs')
    return x,y,pairs


def iter_public_subsets(root):
    for ds in ['HADDAD_OSA_IHH','MARS_IBS_2020','HE_INFANTS_MFGM_2019']:
        x,y,pairs=load_pair(Path(root)/ds)
        if ds=='HADDAD_OSA_IHH':
            for exposure in ['Air','IHH']:
                ids=pairs.loc[pairs.exposure_type.eq(exposure),'subject_id']
                yield ds,exposure,'HADDAD '+exposure,x.loc[ids],y.loc[ids]
        else:
            yield ds,('2_to_12_months' if ds.startswith('HE_') else 'all'),('HE infants 2–12 mo' if ds.startswith('HE_') else 'MARS IBS 2020'),x,y

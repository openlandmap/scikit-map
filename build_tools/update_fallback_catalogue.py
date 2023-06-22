#!/usr/bin/env python
'''
Updates fallback layer info storage with
data from GeoNetwork
'''

import base64
import sys
import gzip
from io import BytesIO
import tempfile
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parent.parent
sys.path = [
    REPO_ROOT.as_posix(),
    *sys.path,
]

import pandas as pd
from skmap.datasets import Catalogue

FALLBACK_FILE = REPO_ROOT/'skmap/datasets/_fallback.py'

if __name__ == '__main__':
    cat = Catalogue(use_csw=True)
    if not cat.use_csw: # abort if unable to connect
        print('GeoNetwork not accessible, aborting update.')
        exit()

    resources = cat.search('')

    layers = pd.DataFrame([
        {
            'url': r,
            **r.meta,
        }
        for r in resources
    ])
    csv = layers.to_csv(index=False)
    csv = gzip.compress(csv.encode())
    csv = base64.b64encode(csv)

    with open(FALLBACK_FILE, 'w') as f:
        f.write(f"""'''
Contains the fallback catalogue CSV as a gzipped, base64 encoded string
'''
CATALOGUE_CSV = {repr(csv)}
""")
    print('Fallback catalogue updated.')

'''
Access to eumap demo datasets hosted in zenodo
'''
import requests
import tarfile
import tempfile
import os, sys
import threading
import time
import re
from functools import reduce
from operator import add
from typing import Union, List
import shutil

from ._defaults import DATA_ROOT_NAME

DATASETS = [
    '4582_spain_landcover_samples.gpkg',
    '4582_spain_rasters.tar.gz',
    '4582_spain_rasters_gapfilled.tar.gz',
    '5606_greece_landcover_samples.gpkg',
    '5606_greece_rasters.tar.gz',
    '5606_greece_rasters_gapfilled.tar.gz',
    '9326_italy_landcover_samples.gpkg',
    '9326_italy_rasters.tar.gz',
    '9326_italy_rasters_gapfilled.tar.gz',
    '9529_croatia_landcover_samples.gpkg',
    '9529_croatia_rasters.tar.gz',
    '9529_croatia_rasters_gapfilled.tar.gz',
    '10636_switzerland_landcover_samples.gpkg',
    '10636_switzerland_rasters.tar.gz',
    '10636_switzerland_rasters_gapfilled.tar.gz',
    '14576_netherlands_landcover_samples.gpkg',
    '14576_netherlands_rasters.tar.gz',
    '14576_netherlands_rasters_gapfilled.tar.gz',
    '14580_netherlands_landcover_samples.gpkg',
    '14580_netherlands_rasters.tar.gz',
    '14580_netherlands_rasters_gapfilled.tar.gz',
    '15560_poland_landcover_samples.gpkg',
    '15560_poland_rasters.tar.gz',
    '15560_poland_rasters_gapfilled.tar.gz',
    '16057_ireland_landcover_samples.gpkg',
    '16057_ireland_rasters.tar.gz',
    '16057_ireland_rasters_gapfilled.tar.gz',
    '22497_sweden_landcover_samples.gpkg',
    '22497_sweden_rasters.tar.gz',
    '22497_sweden_rasters_gapfilled.tar.gz',
    'eu_tilling system_30km.gpkg',
]
ALL = DATASETS

KEYWORDS = reduce(add, map(
    lambda ds_name: re.split(r'[\s_\d\.]+', ds_name),
    DATASETS
))
KEYWORDS = sorted(set(filter(
    lambda kw: kw not in ('', 'km', 'eu', 'system'),
    KEYWORDS
)))

TILES = sorted(set(reduce(add, map(
    lambda ds_name: re.findall(r'\d+_[a-z]+', ds_name),
    DATASETS
))))

_CHUNK_LENGTH = 2**13
_DOWNLOAD_DIR = os.getcwd()
_PROGRESS_INTERVAL = .2 # seconds

def get_datasets(keywords: Union[str, List[str]]='') -> list:
    """
    Get dataset filenames by keyword(s).

    :param keywords: One or more keywords to find datasets by. All recognized keywords are stored in ``eumap.datasets.pilot.KEYWORDS``

    :returns: List of datasets
    :rtype: List[str]

    Examples
    ========

    >>> from eumap.datasets import pilot
    >>>
    >>> print('all datasets:\n', pilot.DATASETS)
    >>> print('all keywords:\n', pilot.KEYWORDS)
    >>>
    >>> datasets = pilot.get_datasets('landcover')
    >>> print('found datasets:\n', datasets)

    """

    if isinstance(keywords, str):
        keywords = [keywords]
    return [*filter(
        lambda ds_name: sum([
            keyword in ds_name
            for keyword in keywords
        ]) == len(keywords),
        ALL
    )]

def _make_download_request(dataset:str) -> requests.Response:
    url = f'https://zenodo.org/record/4265314/files/{dataset}?download=1'
    return requests.get(url, stream=True)

class _DownloadWorker:

    def __init__(self, dataset:str, download_dir: str=_DOWNLOAD_DIR):
        self.done = False
        self.dataset = dataset
        self.download_dir = download_dir
        self.progress = 0
        self.downloaded = 0
        self.size = None
        __, self.tmpfile = tempfile.mkstemp()

    def _unpack(self):
        datapath = os.path.join(self.download_dir, DATA_ROOT_NAME)
        try:
            tile_name = next(filter(
                lambda tile: self.dataset.startswith(tile),
                TILES
            ))
            datapath = os.path.join(datapath, tile_name)
        except StopIteration:
            pass

        lock = threading.Lock()
        lock.acquire()
        if not os.path.isdir(datapath):
            os.makedirs(datapath)
        lock.release()

        if self.dataset.endswith('tar.gz'):
            with tarfile.open(self.tmpfile, "r:gz") as archive:
                archive.extractall(datapath)
            os.remove(self.tmpfile)
        else:
            shutil.move(
                self.tmpfile,
                os.path.join(datapath, self.dataset)
            )

    def _download(self):
        with _make_download_request(self.dataset) as resp:
            resp.raise_for_status()
            self.size = int(resp.headers.get('content-length'))
            with open(self.tmpfile, 'wb') as dst:
                for chunk in resp.iter_content(_CHUNK_LENGTH):
                    dst.write(chunk)
                    dst.flush()
                    self.downloaded += _CHUNK_LENGTH
                    self.progress = (100 * self.downloaded) // self.size
        self._unpack()
        self.done = True

    def start(self):
        self.thread = threading.Thread(target=self._download)
        self.thread.start()

def get_data(datasets: Union[str, list], download_dir: str=_DOWNLOAD_DIR):
    """
    Download dataset(s).

    Files will be stored in an ``eumap_data`` subdirectory within the current working directory. Archives are automatically unpacked.

    :param datasets: One or more datasets to download. If datasets is ``'all'``, all files will be downloaded.

    Examples
    ========

    >>> from eumap.datasets import pilot
    >>>
    >>> datasets = pilot.get_datasets('landcover')
    >>> pilot.get_data(datasets)

    """

    if datasets == 'all':
        datasets = DATASETS
    if isinstance(datasets, str):
        datasets = [datasets]
    for dataset in datasets:
        if dataset not in DATASETS:
            print(f'Dataset {dataset} not available, please choose "all" or one/more of the following:')
            for ds_name in DATASETS:
                print('\t • '+ds_name)
            return

    workers = [
        _DownloadWorker(ds, download_dir)
        for ds in datasets
    ]
    n_workers = len(workers)
    for w in workers:
        w.start()

    while True:
        time.sleep(_PROGRESS_INTERVAL)
        done = sum((w.done for w in workers)) == n_workers
        sizes = [w.size for w in workers]
        if None in sizes:
            print('Starting downloads...', end='\r')
            continue
        total_size = sum(sizes)
        total_download = sum((w.downloaded for w in workers))
        total_progress = (100 * total_download) // total_size
        if total_progress < 100:
            print(
                f'{total_progress}% of {n_workers} downloads / ' \
                f'{round(total_download/2**20, 2)} of {round(total_size/2**20, 2)} MB',
                end='\r',
            )
        else:
            print(f'{round(total_size/2**20, 2)} MB downloaded, unpacking...' + ' '*20, end='\r')
        if done:
            print('\nDownload complete.')
            break

if __name__ == '__main__':
    query = sys.argv[1:]
    if not query:
        query = DATASETS
    else:
        query = get_datasets(query)
    get_data(query)

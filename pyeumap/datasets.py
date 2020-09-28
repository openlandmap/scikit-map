import requests
import zipfile
import tempfile
import os, sys

DATASETS = [
    'croatia_9529',
    'ireland_16057',
    'sweden_22497',
]

DATA_ROOT_NAME = 'pilot_tiles'

_datasets = DATASETS + ['all']

def _make_download_request(dataset:str) -> requests.Response:
    datapath = '/pilot_tiles'
    if dataset != 'all':
        datapath += f'/{dataset}'

    url = f'http://80.56.23.93:5000/fsdownload/webapi/file_download.cgi/{dataset}.zip'
    dlname = f'"{dataset}.zip"'
    urlpath = f'["{datapath}"]'

    return requests.post(
        url,
        headers={
            'Cookie': 'sharing_sid=Dw1ZbveLuz0i7pPnicJLqbzHCSnDjXXC',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'en-US,en;q=0.9',
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        data={
            'api': 'SYNO.FolderSharing.Download',
            'method': 'download',
            'version': '2',
            'mode': 'download',
            'stdhtml': 'false',
            'dlname': dlname,
            'path': urlpath,
            '_sharing_id': '"ztCVujZhz"',
            'codepage': 'enu',
        },
        stream=True,
    )

def _unpack(dataset:str, tmpfile:str, download_dir:str) -> str:
    datapath = download_dir
    if dataset in DATASETS:
        datapath = os.path.join(download_dir, DATA_ROOT_NAME)
    if not os.path.isdir(datapath):
        os.mkdir(datapath)
    with zipfile.ZipFile(tmpfile) as archive:
        archive.extractall(datapath)
    os.remove(tmpfile)

def get_data(dataset:str, download_dir:str=os.getcwd()):
    if dataset not in _datasets:
        print('Dataset not available, please choose one of the following:')
        for _ds in _datasets:
            print('\t • '+_ds)
        return

    with _make_download_request(dataset) as resp:
        resp.raise_for_status()
        downloaded = 0
        __, tmpfile = tempfile.mkstemp()
        with open(tmpfile, 'wb') as dst:
            for chunk in resp.iter_content(2**20):
                dst.write(chunk)
                dst.flush()
                downloaded += len(chunk)
                sys.stdout.write('\r%d MB downloaded'%(downloaded // 2**20))
                sys.stdout.flush()
        print('\nUnpacking...')
        _unpack(dataset, tmpfile, download_dir)
        print('Download complete.')

import requests
import sys
import socket
from pathlib import Path
from .s2tiler import S2Tiler

requests.packages.urllib3.disable_warnings()

fld = Path(__file__).resolve().parent

fn_cert = (fld/'cert.pem').as_posix()

class JobExecutorLocal():

    @classmethod
    def get_new_job(cls, params):        
        return cls(**params)

    def __init__(self, **params):
        self.index = params.pop('index')
        self.params = params
        #print(params)
        self.tiler = S2Tiler(**params)
        self.debug = params['debug']

    def worker(self):
        self.result = self.tiler() == 'OK'
        
    def __call__(self):
        print(f'STARTING: index={self.index}, {self.params["out_folder_prefix"]}, {self.params["tile_name"]}')
        sys.stdout.flush()
        self.worker()
        return self
    
    def submit_job_report(self):
        if self.debug:
            print(f'FINISHED TEST: success={self.result}, report=OK')
            if not self.result:
                log = self.tiler.LOG
                fn = Path(self.params['bucket'])/Path(self.params['out_parent_folder'])/f'{self.params["tile_name"]}.log'
                fn.write_text('\n'.join(log))

            sys.stdout.flush()
        else:
            print(f'FINISHED,  success={self.result}, index={self.index}, {self.params["out_folder_prefix"]}, {self.params["tile_name"]}')
            pass
        
class JobExecutor():
    '''
    Class that fetches jobs, executes it and sends report
    '''

    hostname = socket.gethostname()
    satmos_url = 'https://31.45.233.147:5000/api/get_job_v3'
    satmos_params = { 'apikey': '22e3a9eaabcadb21a3d3ef98e231a10beb7a7bccb8090f1682184d9b55817bfd9144b855738d2868c00fb5266db0230bc9753c61abac2015e0e9532c5ba8bb50',
                     'hostname': hostname}
    satmos_report_url = 'https://31.45.233.147:5000/api/report_job'
           
    @classmethod
    def get_new_job(cls, add_params=None):
        '''
        Fetches new job from server

        :param add_params: additional parameters for job

        :returns JobExecutor instance
        '''
        res = requests.get(cls.satmos_url, cls.satmos_params, verify=False) #fn_cert)
        if res.status_code == 200:
            params = res.json()
            if add_params is not None:
                params.update(add_params)
            return cls(params)
        else:
            return None

    def __init__(self, params):
        '''
        Initialization of instance

        :param params: Parameters of job
        '''
        self.status = params.pop('status')
        if self.status != 'OK':
            print (f'SOMETHING IS WRONG: {self.status}')
        else:
            self.job_id = params.pop('job_id')            
            self.params = params 
            self.tmp_folder = params.get('tmp_folder', None)
            self.data_folder = params.get('data_folder', None)      
            self.tiler = S2Tiler(**self.params)

    def worker(self):
        ''' 
        Run worker and check for errors
        '''
        self.result = self.tiler()=='OK'
        self.log = self.tiler.LOG
        return

    def __call__(self):
        '''
        Call worker for current job.
        '''
        print(f'STARTING: job_id={self.job_id}, {self.tiler.out_folder_name[0]}')
        sys.stdout.flush()
        self.worker()
        return self

    def submit_job_report(self):
        '''
        Submit job report to server
        '''
        result = self.result
        log = self.log

        if self.job_id==-1:
            print(f'FINISHED TEST: success={result}, report=OK, {self.tiler.out_folder_name[0]}')
            sys.stdout.flush()
        else:
            params = self.satmos_params.copy()
            data = {'job_id': self.job_id, 'result': int(result), 'log':'\n'.join(log)}
            res = requests.post(self.satmos_report_url, params=params, json=data, verify = False) #fn_cert)
            res = res.json()
            if res.get('status',None)=='OK':
                print(f'FINISHED: success={result}, report=OK, {self.tiler.out_folder_name[0]}' )
            else:
                print(f'FINISHED: success={result}, report={res.json()}, {self.tiler.out_folder_name[0]}' )
            sys.stdout.flush()
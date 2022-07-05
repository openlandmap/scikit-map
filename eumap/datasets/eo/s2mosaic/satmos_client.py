
import sys
import multiprocessing as mp
import random
import time
import psutil
import shutil
from datetime import datetime
from pathlib import Path
import pandas

import warnings


from .job_executor import JobExecutorLocal, JobExecutor

fld = Path(__file__).resolve().parent

class SatMosClient():
    '''
    Main class for client
    '''
    # Some rule of thumb values
    min_avail_memory = 20   # GB
    min_memory_perproc = 15  #GB
    max_proc_perc = 99
    scene_size = 200    # size of one satelite scene in MB
    scene_count = 140   # max number of scenes for one job    

    def __init__(self, job_scheduler, nworkers=None, sleep_sec=5):
        '''
        job_scheduler - scheduler to use
        nworkers - number of worker processess, default: 2*number of cpus
        sleep_sec - number of seconds to wait 
        '''
        self.job_scheduler = job_scheduler
        if nworkers is None:
            nworkers = mp.cpu_count()*2
        self.nworkers = nworkers
        self.sleep_sec = sleep_sec

        data_folder = job_scheduler.params['data_folder']
        if data_folder is not None:
            self.local_data = True
            self.data_folder = data_folder
       
    def check_local_data(self):    
        '''
        Checking local data usage and deleting old files
        '''
        total, used, free, perc = psutil.disk_usage(self.data_folder)
        free = free /1024/1024  # to MB

        if free < self.scene_count * self.scene_size * self.nworkers:
            # need to delete some old scenes
            data_path = Path(self.data_folder)
            images = sorted(list(map(lambda x: (x, x.stat().st_mtime), data_path.rglob('*.jp2'))), key = lambda x:x[1],reverse=True)
            while free<self.scene_count * self.scene_size * self.nworkers:
                f = images.pop()[0]
                #shutil.rmtree(f)
                f.unlink()
                total, used, free, perc = psutil.disk_usage(self.data_folder)
                free = free /1024/1024

    def run(self):
        '''
        Run the client.
        '''
        nworkers = self.nworkers
        #no_more_jobs = False
        jobs_submitted=[]
        pool = mp.Pool(nworkers, maxtasksperchild=1)

        while True:
            time.sleep(self.sleep_sec)   # wait a few seconds
            n_jobs_submitted = len(jobs_submitted)

            total_memory = psutil.virtual_memory().total / (1024**3)    # GB
            avail_memory = psutil.virtual_memory().available / (1024**3) #GB

            if (n_jobs_submitted<nworkers) \
                and ((n_jobs_submitted+1)*self.min_memory_perproc<total_memory-self.min_avail_memory) \
                and (avail_memory>self.min_avail_memory+self.min_memory_perproc) \
                and (psutil.cpu_percent()<self.max_proc_perc):   #or we can take free memory and proc %

                if self.local_data:
                    self.check_local_data()

                # add new job
                job_exec = self.get_new_job() #JobExecutor.get_new_job(self.add_params)

                if job_exec is not None:
                    jobs_submitted.append(pool.apply_async(job_exec))
                    print(f'New job: avail:{avail_memory} GB, n_jobs: {n_jobs_submitted+1}')
                elif n_jobs_submitted==0:
                    print(f'No jobs for me !!!')
                    return                
            
            # check on all submitted jobs and remove all finished
            jobs_working = []
            for job in jobs_submitted:
                if job.ready():
                    job_res = job.get() # get results
                    job_res.submit_job_report()
                else:
                    jobs_working.append(job)
            jobs_submitted = jobs_working

    def get_new_job(self):
        return self.job_scheduler.get_new_job()

class JobSchedulerLocal():
    '''
    Job scheduler that uses local file 
    '''
    def __init__(self, **params): # mosaic_name, df_s2tiles, from_date, to_date, bucket, debug=False):
        self.params = params
        scenes_csv = params.pop('scenes_csv')
        self.df = pandas.read_csv(scenes_csv)
        self.df.scene_date = pandas.to_datetime(self.df.scene_date).dt.date

        self.from_date = params.pop('from_date').date()
        self.to_date = params.pop('to_date').date()
        if 'tiles' in params:
            self.tiles = params.pop('tiles')
        else:
            self.tiles = self.df.scene_tile_name.unique()        
        self.index = 0

    def get_new_job(self):
        if self.index>=len(self.tiles):
            return None
        else:
            tile_name = self.tiles[self.index]
            self.index = self.index + 1 
            dff = self.df.query("scene_tile_name==@tile_name and scene_date>=@self.from_date and scene_date<=@self.to_date").copy()

            params = dict(index=self.index, satimgs = dff, tile_name=tile_name)
            params.update(self.params)
            return JobExecutorLocal.get_new_job(params)
        
class JobScheduler():
    '''
    Standard scheduler that are server based 
    '''
    debug = False
    def __init__(self, **params):
        self.params = params
        if 'debug' in params:
            self.debug = params['debug']
        self.job_executor = JobExecutor

    def get_new_job(self):
        if self.debug:
            return self.job_executor.get_new_job_amdtr(self.params)
        else:
            return self.job_executor.get_new_job(self.params)


def ghmosaic_production():
    '''
    Production procedure for 30m mosaics
    '''
    data_folder = '/data/data'
    tmp_folder = '/data/tmp'

    mem_per_tile = SatMosClient.min_memory_perproc
    mem = psutil.virtual_memory().available/(1024**3)
    nworkers = min(mp.cpu_count(), int(mem/mem_per_tile))
    print(f'nworkers:{nworkers}')

    job_scheduler = JobScheduler(data_folder=data_folder, tmp_folder=tmp_folder, debug=False)
    client= SatMosClient(job_scheduler, nworkers=nworkers)
    client.run()


if __name__ == '__main__':
    ghmosaic_production()



#%%
import numpy as np
from numba import jit, njit, prange
import rasterio.transform
import concurrent.futures
from itertools import islice


@njit(parallel=True)
def _add_to_subgrid_masked(data, subdata, submask, row1, row2, col1, col2, sgrow1, sgcol1):
    sr=sgrow1
    for r in prange(row1, row2):
        rdata=data[r]
        rsubdata = subdata[sr][sgcol1: sgcol1+col2-col1]
        rsubmask = submask[sr][sgcol1: sgcol1+col2-col1]

        rcdata = rdata[col1:col2]
        rcdata[rsubmask] += rsubdata[rsubmask]

        sr+=1

@njit(parallel=True)
def _add_scalar_to_subgrid_masked(data, scalar, submask, row1, row2, col1, col2, sgrow1, sgcol1):
    sr=0
    for r in prange(row1, row2):
        rdata=data[r]
        rsubmask = submask[sr][sgcol1: sgcol1+col2-col1]

        rcdata = rdata[col1:col2]
        rcdata[rsubmask] += scalar

        sr +=1

@njit(parallel=True)
def _add_to_subgrid_inc1(data, cdata, subdata, submask, row, col, sgrow, sgcol, nrows, ncols):
    d = data[row: row+nrows, col: col+ncols]
    sd = subdata[sgrow: sgrow+nrows, sgcol: sgcol+ncols]
    c = cdata[row: row+nrows, col: col+ncols]

    for r in prange(nrows):
        dr = d[r]
        sdr = sd[r]
        cr = c[r]
        mr = submask[r]
        dr[mr] += sdr[mr]    
        cr[mr] += np.uint8(1)

@njit(parallel=True)
def _update_lmask_clip_v1(mask, lmask, gmask, gmask_clip, sg_ul_row, sg_ul_col):
    h,w = mask.shape
    lmask[:]=0
    lmask[sg_ul_row: sg_ul_row + h, sg_ul_col: sg_ul_col + w] = mask
    gmask_clip[:] = np.logical_and(lmask, gmask)

@njit(parallel=True)
def _update_lmask_clip(mask, lmask, gmask, gmask_clip, sg_ul_row, sg_ul_col):
    h,w = mask.shape
    lmask[:]=0
    gmask_clip[:]=0
    for i in prange(h):
        #lmask[sg_ul_row: sg_ul_row + h, sg_ul_col: sg_ul_col + w] = mask
        tmp_mask = mask[i]
        lmask[sg_ul_row + i][sg_ul_col: sg_ul_col + w] = tmp_mask

        gmask_clip[sg_ul_row + i] [sg_ul_col: sg_ul_col + w] = tmp_mask & gmask[sg_ul_row + i] [sg_ul_col: sg_ul_col + w]
#%%            
class GridSystem:
    def __init__(self, nrows, ncols, transform):
        self.nrows = nrows
        self.ncols = ncols
        self.transform = transform
        self.mask = None

    def set_subgrid(self, sg_nrows, sg_ncols, sg_transform):        
        # xy coordinate upper-left corner
        self.sg_ulxy = sg_transform * (0,0)
        self.sg_brxy = sg_transform * (sg_ncols, sg_nrows)

        # row,col u velikom gridu uperr-left pixela malog grida
        self.sg_ul_col, self.sg_ul_row = map(int, ~self.transform * self.sg_ulxy)
        self.sg_br_col, self.sg_br_row = map(int, ~self.transform * self.sg_brxy)

        # upper left row,col in subgrid after clip with grid
        self.ssg_ul_col = abs(min(self.sg_ul_col,0))
        #self.ssg_ul_col = max(self.sg_ul_col,0)
        self.ssg_ul_row = abs(min(self.sg_ul_row,0))
        #self.ssg_ul_row = max(self.sg_ul_row, 0)


        # correction after clipping
        self.sg_ul_col = max(self.sg_ul_col, 0)
        self.sg_ul_row = max(self.sg_ul_row, 0)
        self.sg_br_col = min(self.sg_br_col, self.ncols)
        self.sg_br_row = min(self.sg_br_row, self.nrows)

        self.sg_ulxy = self.transform * (self.sg_ul_col, self.sg_ul_row)
        self.sg_brxy = self.transform * (self.sg_br_col, self.sg_br_row)

        self.sg_nrows = self.sg_br_row - self.sg_ul_row
        self.sg_ncols = self.sg_br_col - self.sg_ul_col

        self.sg_transform = rasterio.transform.from_origin(self.sg_ulxy[0], self.sg_ulxy[1], self.transform.a, -self.transform.e)

        #self.sg_br_col, self.sg_br_row = [int(c) for c in ~self.transform * self.sg_brxy]
        #self.sg_scalex = (self.sg_br_col - self.sg_ul_col)/sg_ncols

    def set_subgrid_mask(self, mask):        
        self.submask = mask[self.ssg_ul_row: self.ssg_ul_row+self.sg_nrows, self.ssg_ul_col: self.ssg_ul_col+self.sg_ncols].copy()
        #if self.mask is None:
        #    self.mask = np.zeros((self.nrows,self.ncols),dtype=bool)
        #self.mask[:]=False
        #self.mask[self.sg_ul_row: self.sg_ul_row+self.sg_nrows, self.sg_ul_col: self.sg_ul_col+self.sg_ncols] = mask 
        #rows, cols = np.where(mask)
        #self.submask_inds = (cols+self.sg_ul_col) + (rows+self.sg_ul_row)*self.ncols
    
    def get_subgrid_bounds(self):
        return [self.sg_ulxy[0], self.sg_brxy[1], self.sg_brxy[0], self.sg_ulxy[1] ]

    def update_lmask(self, mask, lmask):
        '''
        Updates lmask from mask
        mask -- mask of local grid in local grid shape
        lmask -- mask of local grid in global grid shape
        '''
        lmask[:]=0
        lmask[self.sg_ul_row: self.sg_ul_row+self.sg_nrows, self.sg_ul_col: self.sg_ul_col+self.sg_ncols] = mask

    def update_lmask_clip(self, mask, lmask, gmask, gmask_clip):
        _update_lmask_clip(mask, lmask, gmask, gmask_clip, self.sg_ul_row, self.sg_ul_col)

    def add_to_subgrid_masked(self, data, subdata):
        #data.flat[self.submask_inds] += subdata[self.submask]
        #data[self.mask] += subdata[self.submask]
        #_add_to_subgrid_masked(data, subdata, self.submask, self.sg_ul_row, self.sg_ul_row+self.sg_nrows, self.sg_ul_col, self.sg_ul_col+self.sg_ncols, self.ssg_ul_row, self.ssg_ul_col)
        d = data[self.sg_ul_row: self.sg_br_row, self.sg_ul_col: self.sg_br_col]
        sd = subdata[self.ssg_ul_row: self.ssg_ul_row+self.sg_nrows, self.ssg_ul_col: self.ssg_ul_col+self.sg_ncols]
        d[self.submask] += sd[self.submask]

    def add_scalar_to_subgrid_masked(self, data, scalar):
        #data.flat[self.submask_inds] += scalar
        #data[self.mask] += scalar
        #_add_scalar_to_subgrid_masked(data, scalar, self.submask, self.sg_ul_row, self.sg_ul_row+self.sg_nrows, self.sg_ul_col, self.sg_ul_col+self.sg_ncols, self.ssg_ul_row, self.ssg_ul_col)
        d = data[self.sg_ul_row: self.sg_br_row, self.sg_ul_col: self.sg_br_col]
        d[self.submask] += scalar

    def add_to_subgrid_inc1_masked(self, data, cdata, subdata):
        _add_to_subgrid_inc1(data, cdata, subdata, self.submask, self.sg_ul_row, self.sg_ul_col, self.ssg_ul_row, self.ssg_ul_col, self.sg_nrows, self.sg_ncols)

    def gmask2lmask(self, gmask_clip):
        return gmask_clip[self.sg_ul_row: self.sg_ul_row+self.sg_nrows, self.sg_ul_col: self.sg_ul_col+self.sg_ncols]

    def lind2gind(self, rows, cols): 
        '''
        rows, cols -- array of rows and cols in local grid
        gul_row, gul_col -- global row,col of local uper left pixel
        grd_ncols -- number of columns in global grid
        '''
        return (cols+self.sg_ul_col) + (rows+self.sg_ul_row)*self.ncols

    def gind2lrowscols(self, global_inds):
        grows = global_inds // self.ncols
        gcols = global_inds % self.ncols
        rows = grows - self.sg_ul_row
        cols = gcols - self.sg_ul_col
        return rows, cols

    def ind2rc(self, ind):
        r,c = np.divmod(ind, self.ncols)
        return np.c_[r.reshape(-1,1), c.reshape(-1,1)]

    def rc2ind(self, rows, cols):
        return cols + rows*self.ncols

    def get_neighbours_mask(self, mask_from, mask_in1, mask_in2):
        """
        Returns mask that indicates pixels from mask_in1 and mask_in2 that are 4-neighbours of at least one pixel
        from mask_from
        """
        h,w = mask_from.shape

        mask_n1 = np.zeros_like(mask_from)
        mask_n2 = np.zeros_like(mask_from)

        _get_neighbours_mask(mask_from, mask_in1, mask_in2, mask_n1, mask_n2)

        return mask_n1, mask_n2


    def get_neighbours(self, inds_from, inds_in1, inds_in2):
        """
        Returns indices that are 4-neighbours of inds_from and are in inds_in1 or inds_in2
        """
        rows, cols = self.ind2rc(inds_from).T

        # up
        rows1 = rows -1
        rows1[rows1<0] = 0
        up = self.rc2ind(rows1, cols)
        # right
        cols1 = cols +1
        cols1[cols1>=self.ncols] = self.ncols
        right = self.rc2ind(rows, cols1)
        # down
        rows1 = rows +1
        rows1[rows1>=self.nrows] = self.nrows
        down = self.rc2ind(rows1, cols)
        # left
        cols1 = cols -1
        cols[cols1<0] = 0
        left = self.rc2ind(rows,cols1)

        neighbours = np.unique(np.c_[up,right,down,left])
        neighbours = np.setdiff1d(neighbours, inds_from, assume_unique=True)

        n1 = np.intersect1d(neighbours, inds_in1, assume_unique=True)
        n2 = np.intersect1d(neighbours, inds_in2, assume_unique=True)

        ret1 = self.ind2rc(n1)
        #ret1 = np.c_[ret1[0].reshape(-1,1), ret1[1].reshape(-1,1)]
        ret2 = self.ind2rc(n2)
        #ret2 = np.c_[ret2[0].reshape(-1,1), ret2[1].reshape(-1,1)]
        return ret1,ret2

def tprint(*args, **kwargs):
    from datetime import datetime
    import sys

    print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] ', end='')
    print(*args, **kwargs, flush=True)
    #sys.stdout.flush()

def ttprint(*args, **kwargs):
    from datetime import datetime
    import sys

    print(f'[{datetime.now():%H:%M:%S}] ', end='')
    print(*args, **kwargs, flush=True)
    #sys.stdout.flush()


def ThreadGeneratorLazy(worker, args, max_workers, chunk):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        group = islice(args, chunk)
        futures = {executor.submit(worker, *arg) for arg in group}

        while futures:
            done, futures = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )

            for future in done:
                yield future.result()

            group = islice(args, chunk)

            for arg in group:
                futures.add(executor.submit(worker,*arg))


def ProcessGeneratorLazy(worker, args, max_workers, chunk):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        group = islice(args, chunk)
        futures = {executor.submit(worker, *arg) for arg in group}

        while futures:
            done, futures = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )

            for future in done:
                yield future.result()

            group = islice(args, chunk)

            for arg in group:
                futures.add(executor.submit(worker,*arg))
# %%

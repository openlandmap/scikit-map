#%%
import numpy as np
import numba as nb
from numba import njit, jit, prange
import math
#import numexpr as ne

#%%

@njit(parallel=True)
def cr2xy(transform, vx, vy):
    '''
    transform is tuple with transformation parameters a,b,c,d,e,f
    vx is column (+0.5 for center pixel)
    vy is row (+0.5 for center pixel)
    '''
    sa, sb, sc, sd, se, sf = transform
    return (vx * sa + vy * sb + sc, vx * sd + vy * se + sf)

@njit(parallel=True)
def cr2xy_v2(gmask_clip, transform):
    vy, vx = gmask_clip.nonzero()
    sa, sb, sc, sd, se, sf = transform
    return (vx * sa + vy * sb + sc, vx * sd + vy * se + sf)

@njit(parallel=True)
def cr2xy_v4(gmask_clip, transform):
    sa, sb, sc, sd, se, sf = map(np.float32,transform)

    n = gmask_clip.sum()
    x = np.empty(n, dtype=np.float32) #np.array([], dtype = np.float32)
    y = np.empty(n, dtype=np.float32) #np.array([], dtype = np.float32)
        
    nr = len(gmask_clip)
    xx = [np.empty(0, dtype=np.float32)]*nr
    yy = [np.empty(0, dtype=np.float32)]*nr
    ni = np.zeros(nr, dtype=np.int32)

    for i in prange(nr):
        #print(i)
        vxi = gmask_clip[i].nonzero()[0]
        vyi = np.zeros_like(vxi) + i
        if len(vxi)>0:
            xi = vxi * sa + vyi * sb + sc
            yi = vxi * sd + vyi * se + sf
            xx[i] = xi
            yy[i] = yi
            ni[i] = len(xi)

    nn = 0
    for i in range(nr):
        nii = ni[i]
        ni[i] = nn
        nn += nii

    for i in prange(nr):
        li = len(xx[i])        
        if li>0:
            nni = ni[i]
            x[nni: nni+li] = xx[i]
            y[nni: nni+li] = yy[i]        

    return x,y

# %%
@njit("f8(f8,f8,f8,f8,f8,f8)")
def dist_point_lineseg(x, y, x1, y1, x2, y2):
    '''
    Distance from point (x,y) to line segment (x1,y1):(x2,y2)
    Signed, negative if point is left of line (x coordinate of point is less then nearest point on lineseg)
    '''
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    if (len_sq != 0): #in case of 0 length line
        param = dot / len_sq

    if (param < 0):
        xx = x1
        yy = y1
  
    elif (param > 1):
        xx = x2
        yy = y2

    else:
        xx = x1 + param * C
        yy = y1 + param * D
  
    dx = x - xx
    dy = y - yy
    return np.sqrt(dx * dx + dy * dy) * np.sign(dx)

@njit(parallel=True)
def _distance_to_linestring(x, y, line, dst):
    '''
    parallel numba version of finding distances of multiple points defined with one dimensional arrays x and y 
    to line defined with twodimensional numpy array 
    dst is outpu array with distances
    '''
    for i in prange(len(x)):
        mindist=1e10
        for j in range(len(line)//2):
            d = dist_point_lineseg(x[i], y[i], line[j*2,0], line[j*2,1], line[j*2+1,0], line[j*2+1,1])
            if abs(d)<abs(mindist):
                mindist=d
        dst[i] = mindist
        
@njit
def distance_points2line(x, y, line):
    '''
    returns distance from point (x,y) to line
    x, y -- coordinates of point -- np array m
    line -- np array nx2 -- linestrings
    '''
    dst = np.zeros(x.shape[0], dtype=np.float32)
    _distance_to_linestring(x, y, line, dst)

    return dst


def distance_to_lines(x, y, line1, line2):
    '''
    returns distances from point (x,y) to line1 and line2
    x,y point  -- np array m
    line1, line2  - np array n x 2 -- linestrings
    '''
    dst1 = np.zeros(x.shape[0], dtype=np.float32)
    dst2 = dst1.copy()

    _distance_to_linestring(x, y, line1, dst1)
    _distance_to_linestring(x, y, line2, dst2)

    return dst1, dst2


@njit(parallel=True)
def orbit_average(gdata, cdata, nodata):
    h,w = gdata.shape
    for r in prange(h):
        rcdata=cdata[r]
        rgdata=gdata[r]
        ind = rcdata>0
        rgdata[ind] = (rgdata[ind] + rcdata[ind] // 2)//rcdata[ind] # ISTO KAO g/c+1/2 ali roundano
        rgdata[~ind] = nodata 
    return gdata

def _zvalue_from_index(arr, ind):
    """private helper function to work around the limitation of np.choose() by employing np.take()
    arr has to be a 3D array
    ind has to be a 2D array containing values for z-indicies to take from arr
    See: http://stackoverflow.com/a/32091712/4169585
    This is faster and more memory efficient than using the ogrid based solution with fancy indexing.
    """
    # get number of columns and rows
    nR, nC, nZ = arr.shape

    # get linear indices and extract elements with np.take()
    ii = np.arange(nC*nR).reshape((nC,nR))
    idx = ind + ii%nC * nZ + ii//nC *nZ*nC
    return np.take(arr,idx)

#@jit(nopython=True, parallel=True)
def nan_percentile(arr, q):
    '''
    Equivalent of np.nanpercentile with option interpolation='higher'
    '''
    # valid (non NaN) observations along the (first) third axis
    valid_obs = np.sum(np.isfinite(arr), axis=2)
    
    # sort - former NaNs will move to the end
  
    arr.sort(axis=2) # = np.sort(arr, axis=2)

    # loop over requested quantiles
    if type(q) is list:
        qs = []
        qs.extend(q)
    else:
        qs = [q]

    #if len(qs) < 2:
    #    quant_arr = np.full(shape=(arr.shape[0], arr.shape[1], 1), fill_value=np.nan)
    #else:
    quant_arr = np.full(shape=(len(qs), arr.shape[0], arr.shape[1]), fill_value=np.nan)

    for i in range(len(qs)):
        quant = qs[i]
        # desired position as well as floor and ceiling of it
        c_arr = np.ceil((valid_obs - 1) * (quant / 100.0)).astype(np.int32)
        quant_arr[i, :,:] = _zvalue_from_index(arr=arr, ind=c_arr)
    
    return quant_arr, valid_obs

@njit(parallel=True)
def mosaic_final_weight(dstl, dstg, sdata_clip, gdata_clip, max_distance_from_orbit, dst_dtype, nthreads):
    #height = dstl.shape[0]    
    #for r in prange(height):
    #if nthreads==0:
    #    nthreads = numba.config.NUMBA_NUM_THREADS

    wgmax = np.float32(0)
    wgmin = np.float32(1)
    n = len(dstl)
    w = np.zeros_like(dstg, dtype=np.float32)
    chunk = int(math.ceil(n / nthreads))
    for i in prange(nthreads):
        i_start = i * chunk
        i_end = min(n, (i + 1) * chunk)

        dstgi = dstg[i_start:i_end]
        dstli = dstl[i_start:i_end]
        wg = w[i_start: i_end]

        wg[:] = np.abs(dstgi)/(np.abs(dstli) + np.abs(dstgi))                        
                       
        wg[(wg<0.4) | (dstgi>0) | (dstli>max_distance_from_orbit)] = 0.4
        wg[(wg>0.6) | (dstli<0) | (-dstgi>max_distance_from_orbit)] = 0.6

        wgmax=max(wgmax, wg.max())
        wgmin=min(wgmin, wg.min())

    #wgmax = w.max()
    #wgmin = w.min()
    #print(wgmin, wgmax)

    if (wgmax - wgmin) < 0.05:  # 
        w[:] = 0 if wgmax>0.5 else 1
    else:
        w = (wgmax - w)/(wgmax - wgmin)    

    for i in prange(nthreads):
        i_start = i * chunk
        i_end = min(n, (i + 1) * chunk)

        gdata_clip_i = gdata_clip[i_start: i_end]
        sdata_clip_i = sdata_clip[i_start: i_end]
        wi = w[i_start: i_end]

        gdata_clip_i[:] = (gdata_clip_i * wi + sdata_clip_i * (1-wi)).astype(dst_dtype)


@njit(parallel=True, fastmath=True)
def mosaic_final_weight_v1(dstl, dstg, sdata_clip, gdata_clip, max_distance_from_orbit): #, dst_dtype, nthreads):
    #height = dstl.shape[0]    
    #for r in prange(height):
    #if nthreads==0:
    #    nthreads = numba.config.NUMBA_NUM_THREADS

    wgmax = np.float32(0)
    wgmin = np.float32(1)
    n = len(dstl)
    w = np.zeros_like(dstg, dtype=np.float32)
    #chunk = int(math.ceil(n / nthreads))
    for i in prange(n):
        #i_start = i * chunk
        #i_end = min(n, (i + 1) * chunk)

        dstgi = dstg[i]
        dstli = dstl[i]
        
        wi = abs(dstgi)/(abs(dstli) + abs(dstgi))                        

        if  (wi<0.4) | (dstgi>0) | (dstli>max_distance_from_orbit):
            wi = 0.4
        elif (wi>0.6) | (dstli<0) | (-dstgi>max_distance_from_orbit):
            wi = 0.6

        w[i] = wi
        #wg[(wg<0.4) | (dstgi>0) | (dstli>max_distance_from_orbit)] = 0.4
        #wg[(wg>0.6) | (dstli<0) | (-dstgi>max_distance_from_orbit)] = 0.6

        #wgmax=max(wgmax, wg.max())
        #wgmin=min(wgmin, wg.min())

    wgmax = w.max()
    wgmin = w.min()
    #print(wgmin, wgmax)

    if (wgmax - wgmin) < 0.05:  # 
        w[:] = 0 if wgmax>0.5 else 1
    else:
        w = (wgmax - w)/(wgmax - wgmin)    

    #print(w.min(), w.max())

    for i in prange(n):
        #i_start = i * chunk
        #i_end = min(n, (i + 1) * chunk)

        #gdata_clip_i = gdata_clip[i]
        #sdata_clip_i = sdata_clip[i]
        #wi = w[i]

        gdata_clip[i] = (gdata_clip[i] * w[i] + sdata_clip[i] * (1-w[i]))
    
    #print(gdata_clip.mean())

'''
def mosaic_final_weight_v2(dstl, dstg, sdata_clip, gdata_clip, max_distance_from_orbit):
    md = max_distance_from_orbit
    wg = ne.evaluate('abs(dstg)/(abs(dstl) + abs(dstg))')                        
    wg = ne.evaluate('where((wg<0.4)|(dstg>0)|(dstl>md), 0.4, where((wg>0.6)|(dstl<0)|(-dstg>md),0.6,wg))')                           
    #wg[(wg<0.4) | (dstg>0) | (dstl>self.max_distance_from_orbit)] = 0.4
    #wg[(wg>0.6) | (dstl<0) | (-dstg>self.max_distance_from_orbit)] = 0.6

    wgmax=wg.max()
    wgmin=wg.min()

    if (wgmax - wgmin) < 0.05:  # 
        w = 0 if wgmax>0.5 else 1
    else:
        w = ne.evaluate('(wgmax-wg)/(wgmax-wgmin)')
        #w = (wgmax - wg)/(wgmax - wgmin)    
                    
    mdata = ne.evaluate('gdata_clip * w + sdata_clip * (1-w)')
    return mdata
'''

@njit(parallel=True)
def _update_mask_or(gmask, lmask):
    for i in prange(gmask.shape[0]):
        gmask[i] |= lmask[i]
            
def test_orbit_average():
    '''
    Testing orbit_average
    '''
    import numpy as np
    import numba as nb
    from numba import njit, jit, prange
    from mosaic_helper import ttprint

    w=h=20000
    p=60

    print('Arrays allocation ...')
    gdata=np.zeros((w,h), dtype='int32')
    cdata = np.zeros_like(gdata, dtype='uint8')

    print('Randomize 1...')
    ind = (np.random.rand(len(gdata.flat))*100)<p
    print('Randomize 2...')
    s=ind.sum()
    gdata.reshape(-1)[ind] = np.random.randint(1000,10000,s)
    print('Randomize 3...')
    cdata.reshape(-1)[ind] = np.random.randint(10,100,s)

    ttprint('Numpy ...')
    res_1 = orbit_average_np(gdata.copy(), cdata.copy(), 32000)

    ttprint('Numba ...')
    res_2 = orbit_average_v1(gdata,cdata,32000)
    ttprint('Gotovo'    )
    ttprint(np.isclose(res_1, res_2).all())

if __name__=='__main__':
    test_orbit_average()

# %%


from numpy.typing import NDArray, ArrayLike
from typing import Any, Union

import numpy as np


def get_SWA_weights(att_env, att_seas, season_size, n_imag) -> NDArray:
    # TODO: float32
    conv_mat_row = np.zeros((n_imag))
    base_func = np.zeros((season_size,))
    period_y = season_size/2.0
    slope_y = att_seas/10/period_y
    for i in np.arange(season_size):
        if i <= period_y:
            base_func[i] = -slope_y*i
        else:
            base_func[i] = slope_y*(i-period_y)-att_seas/10
    # Compute the envelop to attenuate temporarly far images
    env_func = np.zeros((n_imag,))
    delta_e = n_imag
    slope_e = att_env/10/delta_e
    for i in np.arange(delta_e):
        env_func[i] = -slope_e*i
        conv_mat_row = 10.0**(np.resize(base_func,n_imag) + env_func)
    return conv_mat_row


def process_image_in_chunks(image, chunk_size, gap_stripes_th, gap_general_th, fft_th):
    mask = np.isnan(image)
    height, width = image.shape
    n_chunk_height = int(np.floor(height/chunk_size))
    n_chunk_width = int(np.floor(width/chunk_size))
    gap_fraq = np.zeros((n_chunk_height, n_chunk_width))
    fft_score = np.zeros((n_chunk_height, n_chunk_width))
    rec_flag = np.zeros((n_chunk_height, n_chunk_width))
    #output_image = image.copy()
    row_starts, row_ends, col_starts, col_ends, fill_true_erase_false = [], [], [], [], []
    
    # Loop through the image by chunks
    for i in range(0, n_chunk_height):
        for j in range(0, n_chunk_width):
            # @FIXME check is also theretically the location of patial frequencies in different share chunks is the same 
            if i != (n_chunk_height-1):
                row_start, row_end = (i * chunk_size, (i+1) * chunk_size)
            else:
                row_start, row_end = (i * chunk_size, height)
            if j != (n_chunk_width-1):
                col_start, col_end = (j * chunk_size, (j+1) * chunk_size)
            else:
                col_start, col_end = (j * chunk_size, width)
            image_chunk = image[row_start:row_end, col_start:col_end]
            mask_chunk = mask[row_start:row_end, col_start:col_end]
            gap_count_chunk = np.sum(mask_chunk)
            gap_fraq[i, j] = gap_count_chunk/(row_end-row_start)/(col_end-col_start)
            if gap_fraq[i, j] < gap_general_th:
                row_starts += [row_start]
                row_ends += [row_end]
                col_starts += [col_start]
                col_ends += [col_end]
                fill_true_erase_false += [True]
                rec_flag[i,j] = 1
            else:
                image_filled = np.nan_to_num(image_chunk, nan=0)
                image_filled = image_filled[0:chunk_size,0:chunk_size].copy()
                # image_filled /= max(np.max(image_filled),1)
                image_filled[image_filled!=0] = 1
                ft = np.fft.ifftshift(image_filled)
                ft = np.fft.fft2(ft, norm='ortho')
                ft = np.fft.fftshift(ft)
                ft[48:80,48:80] = 0
                fft_score[i, j] = np.max(np.abs(ft))
                if fft_score[i, j] > fft_th:
                    row_starts += [row_start]
                    row_ends += [row_end]
                    col_starts += [col_start]
                    col_ends += [col_end]
                    if gap_fraq[i, j] < gap_stripes_th:
                        fill_true_erase_false += [True]
                        rec_flag[i,j] = 1
                    else:
                        fill_true_erase_false += [False]
                        rec_flag[i,j] = -1
                    
    return row_starts, row_ends, col_starts, col_ends, fill_true_erase_false, gap_fraq, fft_score, rec_flag
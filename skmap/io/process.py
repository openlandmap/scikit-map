import os
import time
import warnings
from enum import Enum
from typing import List

from scipy.linalg import matmul_toeplitz

try:
    import gc
    import math
    from abc import ABC, abstractmethod
    from datetime import datetime

    import bottleneck as bn
    import numexpr as ne
    import numpy as np
    import pandas as pd
    import pyfftw
    import scipy.sparse as sparse
    import statsmodels.api as sm
    from dateutil.relativedelta import relativedelta
    from pandas import DataFrame
    from scipy.linalg import circulant, matmul_toeplitz
    from scipy.ndimage import convolve1d
    from scipy.signal import find_peaks
    from scipy.sparse.linalg import splu
    from scipy.special import log1p
    from scipy.stats import theilslopes
    from statsmodels.tsa.seasonal import STL

    from skmap import SKMapGroupRunner, SKMapRunner, parallel
    from skmap.io import RasterData
    from skmap.misc import (
        date_range,
        nan_percentile,
    )

    class Transformer(SKMapGroupRunner, ABC):
        def __init__(self, name: str, verbose: bool = True, temporal=False) -> None:
            super().__init__(verbose=verbose, temporal=temporal)
            self.name = name
            self.name_qa = f"{self.name}{RasterData.TRANSFORM_SEP}qa"

        def _new_info(self, rdata, group, outname, nm):
            info = rdata._info()
            new_info = []

            for index, row in info.iterrows():
                start_dt = row[RasterData.START_DT_COL]
                end_dt = row[RasterData.END_DT_COL]
                raster_file = row[RasterData.PATH_COL]

                name = rdata._set_date(outname, start_dt, end_dt, nm=nm, gr=group)

                new_group = f"{group}.{nm}"

                new_info.append(
                    rdata._new_info_row(
                        raster_file,
                        group=new_group,
                        name=name,
                        dates=[start_dt, end_dt],
                    )
                )

            return new_info

        def run(
            self,
            rdata: RasterData,
            group_list: list,
            ginfo_list: list,
            outname: str = None,
        ):
            if outname is None:
                outname = "skmap_{nm}_{gr}_{dt}"

            new_arrays = []
            new_infos = []

            for group in group_list:
                rdata._active_group = group
                array = rdata._array()

                result = self._run(array)

                if isinstance(result, tuple) and len(result) >= 2:
                    new_array, new_array_qa = result[0], result[1]
                else:
                    new_array, new_array_qa = result, None

                new_info = self._new_info(rdata, group, outname, self.name)
                if new_array_qa is not None:
                    new_info += self._new_info(rdata, group, outname, self.name_qa)
                    new_array = np.concatenate([new_array, new_array_qa], axis=0)

                new_arrays.append(new_array)
                new_infos.append(DataFrame(new_info))

            rdata._active_group = None

            new_array = np.concatenate(new_arrays, axis=0)
            new_info = pd.concat(new_infos)

            return new_array, new_info

        @abstractmethod
        def _run(self, data):
            pass

    class Derivator(SKMapGroupRunner, ABC):
        def __init__(self, verbose: bool = True, temporal=False) -> None:
            super().__init__(verbose=verbose, temporal=temporal)

        def _resize_for_output(self, rdata, n_new_bands):
            # The SharedArray model has no in-place resize; output bands are
            # assembled by the _assemble worker after the job loop.
            return

        def run(
            self,
            rdata: RasterData,
            group_list: str,
            ginfo_list: str,
            outname: str = None,
        ):
            """
            Execute the gapfilling approach.
            """

            kwargs = {
                "rdata": rdata,
                "group_list": group_list,
                "ginfo_list": ginfo_list,
            }
            if outname is not None:
                kwargs["outname"] = outname

            new_array, new_info = self._run(**kwargs)

            return new_array, new_info

        @abstractmethod
        def _run(
            self, rdata: RasterData, group_list: str, ginfo_list: str, outname: str
        ):
            pass

    class Filler(Transformer, ABC):
        def __init__(self, name: str, verbose: bool = True, temporal=False) -> None:
            super().__init__(name=name, verbose=verbose, temporal=temporal)

        def _n_gaps(self, data=None):
            nan_data = np.isnan(data)
            gap_mask = np.logical_not(np.all(nan_data, axis=0))
            return np.sum(nan_data[gap_mask].astype("int"))

        def _run(self, data):
            """
            Execute the gapfilling approach.

            """

            n_gaps = None
            if self.verbose:
                n_gaps = self._n_gaps(data)
                self._verbose(f"There are {n_gaps} gaps in {data.shape}")

            start = time.time()
            result = self._gapfill(data)

            if isinstance(result, tuple) and len(result) >= 2:
                filled = result[0]
            else:
                filled = result

            if self.verbose:
                r_gaps = self._n_gaps(filled)
                gaps_perc = (n_gaps - r_gaps) / n_gaps
                self._verbose(
                    f"{gaps_perc * 100:.2f}% of the gaps filled in {(time.time() - start):.2f} segs"
                )

                if gaps_perc < 1:
                    self._verbose(f"Remained gaps: {r_gaps}")

            return result

        @abstractmethod
        def _gapfill(self, data):
            pass

    class SircleTransformer(Transformer):
        """
        Backend support: ``convolve1d`` (sparse/dense) is jitted on numba and
        falls back to scipy on cpp; ``fft_convolve`` (FFT) uses numpy on all
        backends.

        :param data: N_timeseries x N_samples matrix where the time series are stored one per each row
        :param w_0: convolution coefficent associated with the present
        :param w_f: convolution coefficents associated with the future
        :param w_p: convolution coefficents associated with the past
        :param use_mask: decide if to use a mask for weights renormalization
        :param return_den: in case of usage of the mask will return the denominator matrix in the Hadamard division
        :param S: optional N_timeseries x N_samples matrix where per element scalings are stored
        :param use_fft_backend: force usage of FFT backend computation of the convolution
        :param n_jobs: number of CPU to be used in parallel
        """

        def __init__(
            self,
            wv_0: float,
            wv_f=[],
            wv_p=[],
            wm_0: float = None,
            wm_f=[],
            wm_p=[],
            use_mask: bool = False,
            return_den: bool = False,
            keep_original_values: bool = True,
            S=[],
            conv_backend: str = "dense",
            n_jobs: int = os.cpu_count(),
            verbose=False,
        ) -> None:
            super().__init__(name="SIRCLE", verbose=verbose, temporal=True)
            self.wv_0 = wv_0
            self.wv_f = wv_f
            self.wv_p = wv_p
            self.wm_0 = wm_0
            self.wm_f = wm_f
            self.wm_p = wm_p
            self.use_mask = use_mask
            self.return_den = return_den
            self.keep_original_values = keep_original_values
            self.S = S
            # conv_backend selects the convolution algorithm (dense/sparse/FFT);
            # the ComputeBackend (self.backend) is set by RasterData.run.
            self.conv_backend = conv_backend
            self.n_jobs = n_jobs

        def _run(self, data):
            # Convolution and normalization
            np.seterr(divide="ignore", invalid="ignore")
            orig_shape = data.shape
            data = np.ascontiguousarray(data.T)
            # @TODO avoid this and include the multiband case
            if data.ndim > 1:
                n_t = data.shape[0]
                n_s = data.shape[1]
            else:
                n_t = 1
                n_s = data.size

            assert self.wv_p.ndim == 1, "wv_p must be a 1D array"
            assert self.wv_f.ndim == 1, "wv_f must be a 1D array"
            if self.use_mask:
                if self.wm_0 is None:
                    self.wm_0 = self.wv_0
                if self.wm_p == []:
                    self.wm_p = self.wv_p.copy()
                else:
                    assert self.wv_p.ndim == 1, "wm_p must be a 1D array"
                if self.wm_f == []:
                    self.wm_f = self.wv_f.copy()
                else:
                    assert self.wv_f.ndim == 1, "wm_f must be a 1D array"
                assert self.wm_p.shape == self.wv_p.shape, (
                    "wm_p must be of the same size of wv_p"
                )
                assert self.wm_f.shape == self.wv_f.shape, (
                    "wm_f must be of the same size of wv_f"
                )
            n_p = self.wv_p.size
            n_f = self.wv_f.size
            n_e = n_s + max(n_p, n_f)
            assert max(n_p, n_f) <= n_s, (
                "the size of wv_p and of wv_f should be inferior to the one of the time series"
            )

            V_e = np.zeros((n_t, n_e), dtype=np.float64, order="F")
            V_e[:, 0:n_s] = data
            if self.use_mask:
                valid_mask = ~np.isnan(V_e).astype(bool)
                valid_mask[:, n_s:] = False
                V_e[~valid_mask] = 0.0
                M_e = valid_mask.astype(np.float64)

            if self.conv_backend == "dense":
                # The dense circulant matmul is a zero-padded convolution;
                # route it through the backend FIR op (same maths, no circulant).
                n_pad = max(n_f, n_p)
                wv_e = np.zeros(n_pad * 2 + 1)
                wv_e[n_pad] = self.wv_0
                wv_e[n_pad - n_f : n_pad] = self.wv_f[::-1]
                wv_e[n_pad + 1 : n_pad + 1 + n_p] = self.wv_p[::-1]
                Vt_e = self.backend.convolve1d(V_e[:, 0:n_s], wv_e, axis=-1, mode="constant", cval=0)
                if self.use_mask:
                    wm_e = np.zeros(n_pad * 2 + 1)
                    wm_e[n_pad] = self.wm_0
                    wm_e[n_pad - n_f : n_pad] = self.wm_f[::-1]
                    wm_e[n_pad + 1 : n_pad + 1 + n_p] = self.wm_p[::-1]
                    Mt_e = self.backend.convolve1d(M_e[:, 0:n_s], wm_e, axis=-1, mode="constant", cval=0)

            elif self.conv_backend == "sparse":
                n_pad = max(len(self.wv_f), len(self.wv_p))
                wv_e = np.zeros(n_pad * 2 + 1)
                wv_e[n_pad] = self.wv_0
                wv_e[n_pad - len(self.wv_f) : n_pad] = self.wv_f[::-1]
                wv_e[n_pad + 1 : n_pad + 1 + len(self.wv_p)] = self.wv_p[::-1]
                Vt_e = self.backend.convolve1d(V_e[:, 0:n_s], wv_e, axis=-1, mode="constant", cval=0)
                if self.use_mask:
                    wm_e = np.zeros(n_pad * 2 + 1)
                    wm_e[n_pad] = self.wm_0
                    wm_e[n_pad - len(self.wm_f) : n_pad] = self.wm_f[::-1]
                    wm_e[n_pad + 1 : n_pad + 1 + len(self.wm_p)] = self.wm_p[::-1]
                    Mt_e = self.backend.convolve1d(
                        M_e[:, 0:n_s], wm_e, axis=-1, mode="constant", cval=0
                    )

            elif self.conv_backend == "FFT":
                wv_e = np.zeros((n_e,))
                wv_e[0] = self.wv_0
                wv_e[1 : n_p + 1] = self.wv_p[::-1]
                if n_f > 0:
                    wv_e[-n_f:] = self.wv_f[::-1]
                Vt_e = self.backend.fft_convolve(V_e, wv_e, n_s)
                if self.use_mask:
                    wm_e = np.zeros((n_e,))
                    wm_e[0] = self.wm_0
                    wm_e[1 : n_p + 1] = self.wm_p[::-1]
                    if n_f > 0:
                        wm_e[-n_f:] = self.wm_f[::-1]
                    Mt_e = self.backend.fft_convolve(M_e, wm_e, n_s)

            else:
                raise ValueError("Invalid backend specified")

            if self.use_mask:
                Vt_e = Vt_e / Mt_e
                if self.keep_original_values:
                    Vt_e[valid_mask[:, 0:n_s]] = V_e[valid_mask]
                min_non_zero = np.min(wm_e[wm_e != 0.0])
                assert min_non_zero > pow(10.0, -np.finfo(Mt_e.dtype).precision - 1), (
                    "Use larger values for the non-zero elements of the mask weighting vector, \
          otherise numerical noise could make indistinguishable actual zeros form numerical zeros."
                )
                numerical_zeros_mask = np.abs(Mt_e) < min_non_zero
                Vt_e[numerical_zeros_mask] = np.nan
                if self.return_den:
                    Mt_e[numerical_zeros_mask] = 0.0
                    n_pad = max(len(self.wv_f), len(self.wv_p))
                    wm_e = np.zeros(n_pad * 2 + 1)
                    wm_e[n_pad] = self.wm_0
                    wm_e[n_pad - len(self.wm_f) : n_pad] = self.wm_f[::-1]
                    wm_e[n_pad + 1 : n_pad + 1 + len(self.wm_p)] = self.wm_p[::-1]
                    tmp_vec = np.zeros((n_e,), dtype=np.float64)
                    tmp_vec[0:n_s] = 1.0
                    norm_mask = max(
                        self.backend.convolve1d(tmp_vec, wm_e, axis=-1, mode="constant", cval=0)
                    )
                    Mt_e[:, :n_s] = Mt_e[:, :n_s] / norm_mask
                    # Normalize by the best acheavable weight
                    if self.keep_original_values:
                        Mt_e[valid_mask[:, 0:n_s]] = 1.0
                    return np.reshape(Vt_e.T, orig_shape), np.reshape(Mt_e.T, orig_shape)
                else:
                    return np.reshape(Vt_e.T, orig_shape)
            else:
                return np.reshape(Vt_e.T, orig_shape)

    class SeasConvFill(Filler):
        """
        Backend support: ``tsirf`` is jitted on numba and uses the C++
        ``applyTsirf`` kernel on cpp (float32); the QA path falls back to scipy.

        :param season_size: number of images per year
        :param att_seas: dB of attenuation for images of opposite seasonality
        :param att_env: dB of attenuation for temporarily far images
        :param n_cpu: number of CPU to be used in parallel
        """

        def __init__(
            self,
            season_size: int,
            att_seas: float = 60,
            att_env: float = 20,
            conv_vect_future=[],
            conv_vect_past=[],
            return_qa: bool = False,
            n_jobs: int = os.cpu_count(),
            verbose=False,
        ) -> None:
            super().__init__(name="seasconv", verbose=verbose, temporal=True)
            self.season_size = season_size
            self.return_qa = return_qa
            self.att_seas = att_seas
            self.att_env = att_env
            self.conv_vect_future = conv_vect_future
            self.conv_vect_past = conv_vect_past
            self.n_jobs = n_jobs

        def _compute_conv_mat_row(self, n_imag):
            # Compute a triangular basis function with yaerly periodicity
            conv_mat_row = np.zeros((n_imag))
            base_func = np.zeros((self.season_size,))
            period_y = self.season_size / 2.0
            slope_y = self.att_seas / 10 / period_y
            for i in np.arange(self.season_size):
                if i <= period_y:
                    base_func[i] = -slope_y * i
                else:
                    base_func[i] = slope_y * (i - period_y) - self.att_seas / 10
            # Compute the envelop to attenuate temporarly far images
            env_func = np.zeros((n_imag,))
            delta_e = n_imag
            slope_e = self.att_env / 10 / delta_e
            for i in np.arange(delta_e):
                env_func[i] = -slope_e * i
            conv_mat_row = 10.0 ** (np.resize(base_func, n_imag) + env_func)
            return conv_mat_row

        def _fftw_toeplitz_matmul(self, data, valid_mask):
            tmp_norm_vec = np.ones((data.shape[0], 1))
            tmp_norm_vec[0] = 0.0
            norm_vec = self.backend.toeplitz_matmul(
                self.conv_vect_past, self.conv_vect_future, tmp_norm_vec
            )
            filled = self.backend.toeplitz_matmul(
                self.conv_vect_past, self.conv_vect_future, data
            )
            filled_qa = self.backend.toeplitz_matmul(
                self.conv_vect_past, self.conv_vect_future, valid_mask
            )
            conv_vec = np.concatenate(
                (self.conv_vect_past, self.conv_vect_future[-1:0:-1])
            )
            nz_conv_vec = conv_vec[conv_vec > 0]
            min_conv_val = np.min(nz_conv_vec)
            filled = filled / filled_qa
            no_fill_mask = filled_qa < min_conv_val
            filled_qa /= np.max(norm_vec)
            filled[no_fill_mask] = np.nan
            filled_qa[no_fill_mask] = 0
            return filled, filled_qa

        def _gapfill(self, data):
            np.seterr(divide="ignore", invalid="ignore")
            orig_shape = data.shape
            data = np.ascontiguousarray(data)
            n_imag = data.shape[0]
            if self.season_size * 2 > n_imag:
                warnings.warn(
                    "Less then two years of images available, the time series reconstruction will not take advantage of seasonality"
                )
            half_conv_vect = self._compute_conv_mat_row(n_imag)
            if len(self.conv_vect_future) == 0:
                self.conv_vect_future = half_conv_vect
            if len(self.conv_vect_past) == 0:
                self.conv_vect_past = half_conv_vect

            if self.return_qa:
                # QA needs the convolved mask; the C++ tsirf kernel does not
                # expose QA, so this falls back to the numpy/scipy Toeplitz
                # implementation on every backend.
                valid_mask = ~np.isnan(data)
                data[~valid_mask] = 0.0
                filled, filled_qa = self._fftw_toeplitz_matmul(
                    data, valid_mask.astype(float)
                )
                filled[valid_mask] = data[valid_mask]
                filled_qa[valid_mask] = 1.0
                filled_qa = filled_qa * 100
                filled_qa[filled_qa == 0.0] = np.nan
                return filled, filled_qa

            filled = self.backend.tsirf(
                data, self.conv_vect_past, self.conv_vect_future,
                keep_original_values=True,
            )
            return filled

    class WhittakerSmooth(Transformer):
        """
        https://github.com/mhvwerts/whittaker-eilers-smoother/blob/master/whittaker_smooth.py

        The per-pixel sparse solve uses SciPy (``splu``) on every compute
        backend; only the ``apply_along_axis`` dispatch is backend-aware.
        """

        def __init__(
            self, lmbd=1, d=2, n_jobs: int = os.cpu_count(), verbose=False
        ) -> None:
            super().__init__(name="whittaker", verbose=verbose, temporal=True)

            self.lmbd = lmbd
            self.d = d
            self.n_jobs = n_jobs

        def _speyediff(self, N, d, format="csc"):
            """
            (utility function)
            Construct a d-th order sparse difference matrix based on
            an initial N x N identity matrix

            Final matrix (N-d) x N
            """

            assert not (d < 0), "d must be non negative"
            shape = (N - d, N)
            diagonals = np.zeros(2 * d + 1)
            diagonals[d] = 1.0
            for i in range(d):
                diff = diagonals[:-1] - diagonals[1:]
                diagonals = diff
            offsets = np.arange(d + 1)
            spmat = sparse.diags(diagonals, offsets, shape, format=format)
            return spmat

        def _process_ts(self, data):
            y = data.reshape(-1).copy()
            n_gaps = np.sum((np.isnan(y)).astype("int"))

            if n_gaps == 0:
                r = self.backend.sparse_solve(self.coefmat, y)
                return r
            else:
                return y

        def _run(self, data):
            m = data.shape[0]
            E = sparse.eye(m, format="csc")
            D = self._speyediff(m, self.d, format="csc")
            self.coefmat = E + self.lmbd * D.conj().T.dot(D)

            return self.backend.apply_along_axis(
                self._process_ts, 0, data, n_jobs=self.n_jobs
            )

    class TimeEnum(Enum):
        MONTHLY = 1
        MONTHLY_15P = 2
        MONTHLY_LONGTERM = 3

        BIMONTHLY = 4
        BIMONTHLY_15P = 5
        BIMONTHLY_LONGTERM = 6

        QUARTERLY = 7
        YEARLY = 8

    class TimeAggregate(Derivator):
        """Backend support: reductions and percentiles are jitted on numba and
        use C++ kernels on cpp (float32); ``post_expression`` uses numexpr on
        all backends.
        """
        def __init__(
            self,
            time: list = [TimeEnum.YEARLY, TimeEnum.MONTHLY_LONGTERM],
            operations: List = ["p25", "p50", "p75", "std"],
            rename_operations: dict = {},
            post_expression: str = None,
            date_overlap: bool = False,
            n_jobs: int = os.cpu_count(),
            verbose=False,
        ) -> None:
            super().__init__(verbose=verbose, temporal=True)

            self.time = time
            self.operations = operations
            self.rename_operations = rename_operations
            self.date_overlap = date_overlap
            self.n_jobs = n_jobs

            self.post_expression = post_expression

            self.percs = []
            self.bn_ops = []

            for op in self.operations:
                if op[0] == "p":
                    self.percs.append(int(op[1:]))
                else:
                    method = f"nan{op}"
                    # Validate against the NumpyBackend (the reference); the
                    # actual dispatch happens through self.backend at runtime.
                    if not hasattr(bn, method):
                        raise Exception(
                            f"Operation {method} is invalid, since bottleneck.{method} not exists."
                        )
                    self.bn_ops.append(op)

        def _op_name(self, op):
            if op in self.rename_operations:
                return self.rename_operations[op]
            else:
                return op

        def _aggregate(self, new_idx, ref_array, array_idx, group, tm, dt1, dt2):
            array = parallel.get_shared(ref_array)

            ops = []
            out_slices = []

            for op in self.bn_ops:
                reduce = getattr(self.backend, f"nan{op}")
                out_slices.append(
                    (
                        [new_idx],
                        0,
                        array.shape[1],
                        reduce(array[array_idx, :], axis=0)[np.newaxis, :],
                    )
                )
                new_idx += 1
                ops.append(self._op_name(f"{op}"))

            if len(self.percs) > 0:
                perc_idx = list(range(new_idx, new_idx + len(self.percs)))
                in_array = array[array_idx, :]
                out_slices.append(
                    (
                        perc_idx,
                        0,
                        array.shape[1],
                        self.backend.nanpercentile(in_array, q=self.percs, axis=0),
                    )
                )
                new_idx += len(self.percs)

                for p in self.percs:
                    ops.append(self._op_name(f"p{p}"))

            if self.post_expression is not None:
                out_slices = [
                    (
                        idx,
                        p0,
                        p1,
                        self.backend.evaluate(
                            self.post_expression, local_dict={"new_array": s}
                        ),
                    )
                    for idx, p0, p1, s in out_slices
                ]

            return (group, ops, tm, dt1, dt2, out_slices)

        def _args_monthly(
            self, rdata, group, start_dt, end_dt, date_format, months=1, daysp=None
        ):
            args = []
            ref_array = rdata.array.ref

            for dt1, dt2 in date_range(
                f"{start_dt.year}0101",
                f"{end_dt.year}1201",
                "months",
                months,
                return_str=True,
                ignore_29feb=True,
                date_format=date_format,
            ):
                dt1a, dt2a = dt1, dt2
                if daysp is not None:
                    dt1a = datetime.strptime(dt1, date_format)
                    dt2a = datetime.strptime(dt2, date_format)
                    dt1a = (dt1a - relativedelta(days=daysp)).strftime(date_format)
                    dt2a = (dt2a + relativedelta(days=daysp)).strftime(date_format)

                tm = ""
                array_idx = rdata.filter_date(
                    dt1a,
                    dt2a,
                    return_idx=True,
                    date_format=date_format,
                    date_overlap=self.date_overlap,
                )

                if len(array_idx) > 0:
                    args += [
                        (
                            ref_array,
                            array_idx,
                            group,
                            tm,
                            datetime.strptime(dt1, date_format),
                            datetime.strptime(dt2, date_format),
                        )
                    ]

            return args

        def _args_yearly(self, rdata, group, start_dt, end_dt, date_format):
            args = []
            ref_array = rdata.array.ref

            for dt1, dt2 in date_range(
                f"{start_dt.year}0101",
                f"{end_dt.year}1201",
                "years",
                1,
                return_str=True,
                ignore_29feb=False,
                date_format=date_format,
            ):
                tm = "yearly"
                array_idx = rdata.filter_date(
                    dt1,
                    dt2,
                    return_idx=True,
                    date_format=date_format,
                    date_overlap=self.date_overlap,
                )

                if len(array_idx):
                    args += [
                        (
                            ref_array,
                            array_idx,
                            group,
                            tm,
                            datetime.strptime(dt1, date_format),
                            datetime.strptime(dt2, date_format),
                        )
                    ]

            return args

        def _args_monthly_longterm(self, rdata, group, start_dt, end_dt, date_format):
            args = []
            ref_array = rdata.array.ref

            for month in range(1, 13):
                array_idx_list = []
                month = str(month).zfill(2)

                for dt1, dt2 in date_range(
                    f"{start_dt.year}{month}01",
                    f"{end_dt.year}{month}01",
                    "months",
                    1,
                    date_offset=11,
                    return_str=True,
                    ignore_29feb=False,
                    date_format=date_format,
                ):
                    array_idx = rdata.filter_date(
                        dt1,
                        dt2,
                        return_idx=True,
                        date_format=date_format,
                        date_overlap=self.date_overlap,
                    )

                    if len(array_idx):
                        array_idx_list += array_idx

                tm = f"m{month}"
                if len(array_idx_list) > 0:
                    # args += [ (np.concatenate(in_array, axis=-1), tm, start_dt, end_dt) ]
                    args += [(ref_array, array_idx_list, group, tm, start_dt, end_dt)]

            return args

        def _run(
            self,
            rdata: RasterData,
            group_list: list,
            ginfo_list: list,
            outname: str = "skmap_aggregate.{gr}_{op}_{dt}",
        ):
            args = []

            for group, ginfo in zip(group_list, ginfo_list):
                date_format = "%Y%m%d"
                start_dt = ginfo[RasterData.START_DT_COL].min()
                end_dt = ginfo[RasterData.END_DT_COL].max()

                rdata._active_group = group

                for t in self.time:
                    if t == TimeEnum.MONTHLY_LONGTERM:
                        args += self._args_monthly_longterm(
                            rdata, group, start_dt, end_dt, date_format
                        )
                    elif t == TimeEnum.YEARLY:
                        args += self._args_yearly(
                            rdata, group, start_dt, end_dt, date_format
                        )
                    elif t == TimeEnum.MONTHLY:
                        args += self._args_monthly(
                            rdata, group, start_dt, end_dt, date_format, 1
                        )
                    elif t == TimeEnum.MONTHLY_15P:
                        args += self._args_monthly(
                            rdata, group, start_dt, end_dt, date_format, 1, 15
                        )
                    elif t == TimeEnum.BIMONTHLY:
                        args += self._args_monthly(
                            rdata, group, start_dt, end_dt, date_format, 2
                        )
                    elif t == TimeEnum.BIMONTHLY_15P:
                        args += self._args_monthly(
                            rdata, group, start_dt, end_dt, date_format, 2, 15
                        )
                    elif t == TimeEnum.QUARTERLY:
                        args += self._args_monthly(
                            rdata, group, start_dt, end_dt, date_format, 3
                        )
                    else:
                        raise Exception(f"Aggregation by {t} not implemented")

            n_new_rasters = len(args) * len(self.operations)
            idx_offset = rdata._idx_offset()
            new_shape = (idx_offset + n_new_rasters, rdata.array.shape[1])
            ref_in = rdata.array.ref
            args = [(ref_in, *arg[1:]) for arg in args]

            _args = []
            for idx, arg in zip(range(0, n_new_rasters, len(self.operations)), args):
                _arg = list(arg)
                _arg.insert(0, idx_offset + idx)
                _args.append(tuple(_arg))

            args = _args
            new_info = []

            self._verbose(
                f"Computing {len(args)} "
                + f"time aggregates from {start_dt.year} to {end_dt.year}"
            )

            specs = []
            for group, ops, tm, dt1, dt2, out_slices in parallel.job(
                self._aggregate, args
            ):
                specs.extend(out_slices)
                for op in ops:
                    _group = group
                    if tm != "":
                        _group = f"{group}.{tm}"

                    rdata._active_group = group
                    name = rdata._set_date(outname, dt1, dt2, op=op, gr=_group)

                    new_group = f"{_group}.{op}"

                    new_info.append(
                        rdata._new_info_row(
                            rdata.base_raster,
                            name=name,
                            group=new_group,
                            dates=[dt1, dt2],
                        )
                    )

            rdata._active_group = None

            out_ref = parallel._remote(
                parallel._assemble, [ref_in], new_shape, specs, idx_offset
            )
            rdata.array = parallel.SharedArray(
                out_ref, new_shape, rdata.array.dtype
            )

            return None, DataFrame(new_info)

    class PeakAnalysis(Derivator):
        """Per-pixel statistics use SciPy/statsmodels on every compute backend
        (``find_peaks``/``theilslopes``/``STL``+``OLS``); only the ``evaluate``
        and ``apply_along_axis`` dispatch is backend-aware.
        """
        def __init__(
            self,
            season_size: int,
            min_height: float = 0.5,
            min_prominence: float = 0.2,
            min_distance: float = 1.0,
            scale_expr: str = None,
            n_jobs: int = os.cpu_count(),
            verbose=False,
        ) -> None:
            super().__init__(verbose=verbose, temporal=True)

            self.season_size = season_size
            self.min_height = min_height
            self.min_prominence = min_prominence
            self.min_distance = min_distance
            self.scale_expr = scale_expr
            self.n_jobs = n_jobs

            self.name_misc = [
                ("peaks", "m", 100),
                ("peaks", "n", 1),
            ]

            self.scale_arr = np.array([scale for _, _, scale in self.name_misc])

        def _find_peaks(self, data):
            if self.scale_expr is not None:
                data = self.backend.evaluate(self.scale_expr, {"data": data})

            has_nan = np.sum(np.isnan(data).astype("int"))

            ts_size = data.shape[0]
            idxs = [
                (i, i + self.season_size) for i in range(0, ts_size, self.season_size)
            ]

            result = np.empty((len(idxs) * 2))

            if has_nan == 0:
                peaks, _ = self.backend.find_peaks(
                    data,
                    height=self.min_height,
                    prominence=self.min_prominence,
                    distance=self.min_distance,
                )
                _peaks = list(peaks)

                o2 = 0

                if len(peaks) > 0:
                    for i0, i1 in idxs:
                        seas_peaks = list((i for i in range(i0, i1) if i in _peaks))
                        nos = len(seas_peaks)

                        mean, los = np.nan, 0
                        if nos > 0:
                            mean = np.mean(data[seas_peaks])
                            los = np.sum(data[i0:i1] > mean * 0.5) / self.season_size

                        result[o2] = los * self.scale_arr[0]
                        result[o2 + 1] = nos * self.scale_arr[1]
                        o2 += 2

            return result

        def _unpack(self, p0, p1, i2, ref_array, idx_offset):
            array = parallel.get_shared(ref_array)
            result = self.backend.apply_along_axis(self._find_peaks, 0, array[i2, p0:p1])
            o2 = list(range(idx_offset, idx_offset + result.shape[0]))
            return (o2, p0, p1, result)

        def _args(self, rdata, ginfo):
            ref_array = rdata.array.ref
            max_pixels = rdata.array.shape[1]
            pixels_per_job = math.ceil(max_pixels / self.n_jobs)

            idx_offset = rdata._idx_offset()

            args = []
            for i in range(0, max_pixels, pixels_per_job):
                p0, p1 = i, (i + pixels_per_job)
                if p1 > max_pixels:
                    p1 = max_pixels

                i2 = ginfo.index
                args.append((p0, p1, i2, ref_array, idx_offset))

            return args

        def _run(
            self,
            rdata: RasterData,
            group_list: list,
            ginfo_list: list,
            outname: str = "skmap_{gr}.{nm}_{pr}_{dt}",
        ):
            new_info = []

            for group, ginfo in zip(group_list, ginfo_list):
                rdata._active_group = group

                start_dt_min = ginfo[RasterData.START_DT_COL].min()
                end_dt_max = ginfo[RasterData.END_DT_COL].max()

                ts_size = ginfo.shape[0]

                n_seasons = ts_size // self.season_size
                idx_offset = rdata._idx_offset()
                n_new = len(self.name_misc) * n_seasons
                new_shape = (idx_offset + n_new, rdata.array.shape[1])
                ref_in = rdata.array.ref

                args = self._args(rdata, ginfo)

                specs = []
                for spec in parallel.job(
                    self._unpack,
                    args,
                    n_jobs=self.n_jobs,
                ):
                    specs.append(spec)

                out_ref = parallel._remote(
                    parallel._assemble, [ref_in], new_shape, specs, idx_offset
                )
                rdata.array = parallel.SharedArray(
                    out_ref, new_shape, rdata.array.dtype
                )

                for i in range(0, ts_size, self.season_size):
                    _i = int(i / self.season_size)
                    i0, i1 = (i, i + self.season_size - 1)

                    start_dt_min = ginfo.iloc[i0][RasterData.START_DT_COL]
                    end_dt_max = ginfo.iloc[i1][RasterData.END_DT_COL]

                    for j, (nm, pr, _) in zip(
                        range(_i, _i + len(self.name_misc)), self.name_misc
                    ):
                        name = rdata._set_date(
                            outname, start_dt_min, end_dt_max, nm=nm, pr=pr, gr=group
                        )

                        new_group = f"{group}.{nm}.{pr}"

                        new_info.append(
                            rdata._new_info_row(
                                rdata.base_raster,
                                group=new_group,
                                name=name,
                                dates=[start_dt_min, end_dt_max],
                            )
                        )

            return None, DataFrame(new_info)

    class SlopeAnalysis(Derivator):
        """Per-pixel statistics use SciPy/statsmodels on every compute backend
        (``find_peaks``/``theilslopes``/``STL``+``OLS``); only the ``evaluate``
        and ``apply_along_axis`` dispatch is backend-aware.
        """
        def __init__(
            self,
            scale_expr: str = None,
            scaling: float = 1.0,
            n_jobs: int = os.cpu_count(),
            verbose=False,
        ) -> None:
            super().__init__(verbose=verbose, temporal=True)

            self.scale_expr = scale_expr
            self.scaling = scaling
            self.n_jobs = n_jobs

        def _theil_slopes(self, data):
            if self.scale_expr is not None:
                data = self.backend.evaluate(self.scale_expr, {"data": data})

            has_nan = np.sum(np.isnan(data).astype("int"))

            result = np.empty(
                1,
            )

            if has_nan == 0:
                result[0], _, _, _ = self.backend.theilslopes(data, np.arange(0, data.shape[0]))
            else:
                result[0] = np.nan
            result[0] *= self.scaling
            return result

        def _unpack(self, p0, p1, i2, ref_array, idx_offset):
            array = parallel.get_shared(ref_array)
            result = self.backend.apply_along_axis(self._theil_slopes, 0, array[i2, p0:p1])
            o2 = list(range(idx_offset, idx_offset + result.shape[0]))
            return (o2, p0, p1, result)

        def _args(self, rdata, ginfo):
            ref_array = rdata.array.ref
            max_pixels = rdata.array.shape[1]
            pixels_per_job = math.ceil(max_pixels / self.n_jobs)

            idx_offset = rdata._idx_offset()

            args = []
            for i in range(0, max_pixels, pixels_per_job):
                p0, p1 = i, (i + pixels_per_job)
                if p1 > max_pixels:
                    p1 = max_pixels

                i2 = ginfo.index
                args.append((p0, p1, i2, ref_array, idx_offset))

            return args

        def _run(
            self,
            rdata: RasterData,
            group_list: list,
            ginfo_list: list,
            outname: str = "{gr}.{nm}_{pr}_{dt}",
        ):
            new_info = []

            for group, ginfo in zip(group_list, ginfo_list):
                rdata._active_group = group

                start_dt_min = ginfo[RasterData.START_DT_COL].min()
                end_dt_max = ginfo[RasterData.END_DT_COL].max()

                idx_offset = rdata._idx_offset()
                new_shape = (idx_offset + 1, rdata.array.shape[1])
                ref_in = rdata.array.ref

                args = self._args(rdata, ginfo)

                specs = []
                for spec in parallel.job(
                    self._unpack,
                    args,
                    n_jobs=self.n_jobs,
                ):
                    specs.append(spec)

                out_ref = parallel._remote(
                    parallel._assemble, [ref_in], new_shape, specs, idx_offset
                )
                rdata.array = parallel.SharedArray(
                    out_ref, new_shape, rdata.array.dtype
                )

                nm = "theilslopes"
                pr = "m"
                name = rdata._set_date(
                    outname, start_dt_min, end_dt_max, nm=nm, pr=pr, gr=group
                )

                new_group = f"{group}.{nm}.{pr}"

                new_info.append(
                    rdata._new_info_row(
                        rdata.base_raster,
                        group=new_group,
                        name=name,
                        dates=[start_dt_min, end_dt_max],
                    )
                )

            return None, DataFrame(new_info)

    class FindMinMax(Derivator):
        """Backend support: ``seasonal_min_max`` is jitted on numba and falls
        back to numpy on cpp; ``scale_expr`` uses numexpr on all backends.
        """
        def __init__(
            self,
            scale_expr: str = None,
            season_size: int = None,
            scaling: float = 1.0,
            min_max: str = None,
            n_jobs: int = os.cpu_count(),
            verbose=False,
        ) -> None:
            super().__init__(verbose=verbose, temporal=True)

            self.scale_expr = scale_expr
            self.season_size = season_size
            self.min_max = min_max
            self.scaling = scaling
            self.n_jobs = n_jobs

        def _find_min_max(self, data):
            if self.scale_expr is not None:
                data = self.backend.evaluate(self.scale_expr, {"data": data})

            ts_size = data.shape[0]

            idxs = [
                (i, i + self.season_size) for i in range(0, ts_size, self.season_size)
            ]
            result = np.empty(len(idxs))

            for j, (i0, i1) in enumerate(idxs):
                has_nan = np.sum(np.isnan(data[i0:i1]).astype("int"))
                if has_nan == 0:
                    if self.min_max == "min":
                        result[j] = np.min(data[i0:i1])
                    elif self.min_max == "max":
                        result[j] = np.max(data[i0:i1])
                    else:
                        assert False, "min_max can be ether 'min' or 'max'"
                else:
                    result[j] = np.nan
                result[j] *= self.scaling
            return result

        def _unpack(self, p0, p1, i2, ref_array, idx_offset):
            array = parallel.get_shared(ref_array)
            sub = array[i2, p0:p1].T
            if self.scale_expr is not None:
                sub = self.backend.evaluate(self.scale_expr, {"data": sub})
            result = self.backend.seasonal_min_max(
                sub, self.season_size, self.min_max, self.scaling
            )
            o2 = list(range(idx_offset, idx_offset + result.shape[1]))
            return (o2, p0, p1, result.T)

        def _args(self, rdata, ginfo):
            ref_array = rdata.array.ref
            max_pixels = rdata.array.shape[1]
            pixels_per_job = math.ceil(max_pixels / self.n_jobs)

            idx_offset = rdata._idx_offset()

            args = []
            for i in range(0, max_pixels, pixels_per_job):
                p0, p1 = i, (i + pixels_per_job)
                if p1 > max_pixels:
                    p1 = max_pixels

                i2 = ginfo.index
                args.append((p0, p1, i2, ref_array, idx_offset))

            return args

        def _run(
            self,
            rdata: RasterData,
            group_list: list,
            ginfo_list: list,
            outname: str = "{gr}.{nm}_{pr}_{dt}",
        ):
            new_info = []

            for group, ginfo in zip(group_list, ginfo_list):
                rdata._active_group = group

                start_dt_min = ginfo[RasterData.START_DT_COL].min()
                end_dt_max = ginfo[RasterData.END_DT_COL].max()

                ts_size = ginfo.shape[0]

                idx_offset = rdata._idx_offset()
                n_new = ts_size // self.season_size
                new_shape = (idx_offset + n_new, rdata.array.shape[1])
                ref_in = rdata.array.ref

                args = self._args(rdata, ginfo)

                specs = []
                for spec in parallel.job(
                    self._unpack,
                    args,
                    n_jobs=self.n_jobs,
                ):
                    specs.append(spec)

                out_ref = parallel._remote(
                    parallel._assemble, [ref_in], new_shape, specs, idx_offset
                )
                rdata.array = parallel.SharedArray(
                    out_ref, new_shape, rdata.array.dtype
                )
                nm = self.min_max
                pr = "m"

                for i0, i1 in [
                    (i, i + self.season_size - 1)
                    for i in range(0, ts_size, self.season_size)
                ]:
                    start_dt_min = ginfo.iloc[i0][RasterData.START_DT_COL]
                    end_dt_max = ginfo.iloc[i1][RasterData.END_DT_COL]
                    name = rdata._set_date(
                        outname, start_dt_min, end_dt_max, nm=nm, pr=pr, gr=group
                    )

                    new_group = f"{group}.{nm}"

                    new_info.append(
                        rdata._new_info_row(
                            rdata.base_raster,
                            group=new_group,
                            name=name,
                            dates=[start_dt_min, end_dt_max],
                        )
                    )
            return None, DataFrame(new_info)

    class TrendAnalysis(Derivator):
        """Per-pixel statistics use SciPy/statsmodels on every compute backend
        (``find_peaks``/``theilslopes``/``STL``+``OLS``); only the ``evaluate``
        and ``apply_along_axis`` dispatch is backend-aware.
        """
        def __init__(
            self,
            season_size: int,
            season_smoother: int = None,
            trend_smoother: int = None,
            log_rescale: tuple = None,
            scale_factor: int = 10000,
            n_jobs: int = os.cpu_count(),
            verbose=False,
        ) -> None:
            super().__init__(verbose=verbose, temporal=True)

            self.season_size = season_size
            self.season_smoother = season_smoother
            self.trend_smoother = trend_smoother
            self.n_jobs = n_jobs

            self.vmin, self.vmax = None, None
            if log_rescale is not None:
                self.vmin, self.vmax = log_rescale

            self.name_misc = [
                ("alpha", "m", scale_factor),
                ("alpha", "sd", scale_factor),
                ("alpha", "tv", scale_factor),
                ("alpha", "pv", scale_factor),
                ("beta", "m", scale_factor),
                ("beta", "sd", scale_factor),
                ("beta", "tv", scale_factor),
                ("beta", "pv", scale_factor),
                ("r2", "m", 100),
            ]

            if self.season_smoother is None:
                self.season_smoother = self.season_size + 1
            if self.trend_smoother is None:
                self.trend_smoother = (2 * self.season_size) + 1

        def _trend_regression(self, data):
            has_nan = np.sum(np.isnan(data).astype("int"))

            ts_size = data.shape[0]
            out_size = ts_size + 9  # fixed number of ols return values

            if has_nan == 0:
                if np.std(data) == 0:
                    nan_result = np.empty(out_size)
                    nan_result[:] = np.nan
                    return nan_result

                res = self.backend.stl_decompose(
                    data.copy(),
                    period=self.season_size,
                    seasonal=self.season_smoother,
                    trend=self.trend_smoother,
                    robust=True,
                )

                y = res.trend

                if self.vmin is not None:
                    y[y > self.vmax] = self.vmax
                    y[y < self.vmin] = self.vmin
                    y = log1p(y / self.vmax)

                y_size = y.shape[0]
                X = np.array(range(0, y_size)) / y_size

                X = sm.add_constant(X)
                results = self.backend.ols(y, X)

                result_stack = np.stack(
                    [results.params, results.bse, results.tvalues, results.pvalues],
                    axis=1,
                )

                return np.concatenate(
                    [
                        res.trend,
                        result_stack[0, :],
                        result_stack[1, :],
                        np.stack([results.rsquared]),
                    ]
                )

            else:
                nan_result = np.empty(out_size)
                nan_result[:] = np.nan
                return nan_result

        def _run(
            self,
            rdata: RasterData,
            group_list: list,
            ginfo_list: list,
            outname: str = "skmap_{gr}.{nm}_{pr}_{dt}",
        ):
            new_arrays = []
            new_infos = []

            for group, ginfo in zip(group_list, ginfo_list):
                rdata._active_group = group
                array = rdata._array()
                info = rdata._info()

                start_dt_min = ginfo[RasterData.START_DT_COL].min()
                end_dt_max = ginfo[RasterData.END_DT_COL].max()

                new_array = self.backend.apply_along_axis(
                    self._trend_regression, axis=0, arr=array, n_jobs=self.n_jobs
                )

                new_info = []

                for index, row in info.iterrows():
                    start_dt = row[RasterData.START_DT_COL]
                    end_dt = row[RasterData.END_DT_COL]

                    nm, pr = ("trend", "m")

                    name = rdata._set_date(
                        outname, start_dt, end_dt, nm=nm, pr=pr, gr=group
                    )

                    new_group = f"{group}.{nm}.{pr}"

                    new_info.append(
                        rdata._new_info_row(
                            rdata.base_raster,
                            group=new_group,
                            name=name,
                            dates=[start_dt, end_dt],
                        )
                    )

                ts_size = array.shape[0]

                for i, (nm, pr, scale) in zip(
                    range(0, len(self.name_misc)), self.name_misc
                ):
                    new_array[ts_size + i, :] *= scale

                    name = rdata._set_date(
                        outname, start_dt_min, end_dt_max, nm=nm, pr=pr, gr=group
                    )

                    new_group = f"{group}.{nm}.{pr}"

                    new_info.append(
                        rdata._new_info_row(
                            rdata.base_raster,
                            group=new_group,
                            name=name,
                            dates=[start_dt_min, end_dt_max],
                        )
                    )

                new_arrays.append(new_array)
                new_infos.append(DataFrame(new_info))

            rdata._active_group = None
            return np.concatenate(new_arrays, axis=0), pd.concat(new_infos)

    class Calc(SKMapRunner):
        """Backend support: ``evaluate`` uses numexpr on all backends (numba
        cannot parse expression strings and cpp has no expression VM).
        """
        def __init__(
            self,
            expressions: dict,
            mask_group: str = None,
            mask_values: list = [],
            n_jobs: int = os.cpu_count(),
            verbose=False,
        ) -> None:
            self.n_jobs = n_jobs

            self.expressions = expressions
            self.mask_group = mask_group
            self.mask_values = mask_values
            self.date_cols = [RasterData.START_DT_COL, RasterData.END_DT_COL]

        def _map(self, ref_array, gmap, new_gmap):
            array_dict = {}
            array = parallel.get_shared(ref_array)

            array_mask = None
            if self.mask_group is not None and len(self.mask_values) >= 1:
                idx = gmap[self.mask_group]
                array_mask = np.isin(array[idx, :], self.mask_values)

            for group in gmap.keys():
                idx = gmap[group]
                array_dict[group] = array[idx, :]
                if array_mask is not None and group != self.mask_group:
                    array_dict[group][array_mask] = np.nan

            out_slices = []
            for group in self.expressions.keys():
                expression = self.expressions[group]
                if group in gmap:
                    idx = gmap[group]
                else:
                    idx = new_gmap[group]
                out_slices.append(
                    (
                        [idx],
                        0,
                        array.shape[1],
                        self.backend.evaluate(expression, local_dict=array_dict),
                    )
                )

            fidx = list(gmap.values())[0]

            return fidx, out_slices

        def run(self, rdata: RasterData, outname: str = "skmap_{gr}_{dt}"):
            self.groups = list(rdata.info[RasterData.GROUP_COL].unique())

            self.new_groups = []
            for key in self.expressions.keys():
                if key not in self.groups:
                    self.new_groups.append(key)

            n_new_groups = len(self.new_groups)
            # Count the date groups so we can resize the array up front.
            n_date_groups = len(list(rdata.info.groupby(self.date_cols)))
            n_new_rasters = n_new_groups * n_date_groups

            idx_offset = rdata._idx_offset()
            new_shape = (idx_offset + n_new_rasters, rdata.array.shape[1])
            ref_in = rdata.array.ref

            args = []

            ref_array = rdata.array.ref

            idx_counter = 0
            for _, rows in rdata.info.groupby(self.date_cols):
                gidx = rows.index
                ggroup = list(rdata.info.iloc[gidx]["group"])

                gmap = {}
                new_gmap = {}

                for idx, group in zip(gidx, ggroup):
                    gmap[group] = idx

                new_group_offset = idx_offset + (idx_counter * n_new_groups)
                for idx, new_group in zip(range(0, n_new_groups), self.new_groups):
                    new_gmap[new_group] = new_group_offset + idx

                args.append((ref_array, gmap, new_gmap))
                idx_counter += 1

            new_info = []
            specs = []

            for fidx, out_slices in parallel.job(
                self._map,
                args,
                n_jobs=self.n_jobs,
            ):
                specs.extend(out_slices)
                row = rdata.info.iloc[fidx]

                start_dt, end_dt = (
                    row[RasterData.START_DT_COL],
                    row[RasterData.END_DT_COL],
                )
                group = row[RasterData.GROUP_COL]

                date_format = rdata.date_args[group]["date_format"]
                date_style = rdata.date_args[group]["date_style"]

                for new_group in self.new_groups:
                    name = rdata._set_date(
                        outname,
                        start_dt,
                        end_dt,
                        date_format=date_format,
                        date_style=date_style,
                        gr=new_group,
                    )
                    new_info.append(
                        rdata._new_info_row(
                            rdata.base_raster,
                            date_format=date_format,
                            date_style=date_style,
                            group=new_group,
                            name=name,
                            dates=[start_dt, end_dt],
                        )
                    )

            out_ref = parallel._remote(
                parallel._assemble, [ref_in], new_shape, specs, idx_offset
            )
            rdata.array = parallel.SharedArray(
                out_ref, new_shape, rdata.array.dtype
            )

            return None, DataFrame(new_info)

        def _calc(self, array_dict):
            if self.mask_group is not None and len(self.mask_values) >= 1:
                array_mask = np.isin(array_dict[self.mask_group], self.mask_values)

                for g in array_dict.keys():
                    if g != self.mask_group:
                        array_dict[g][array_mask] = np.nan

            for group in self.expressions.keys():
                expression = self.expressions[group]
                array_dict[group] = self.backend.evaluate(expression, local_dict=array_dict)

            return array_dict

    class Prediction(SKMapRunner):
        """Predict on a RasterData for **all years at once**, appending the
        result bands to ``rdata.array``.

        All available years are concatenated into a single feature matrix
        (``(n_years·n_pixels, n_features)``) with static covariates repeated
        for every year, and the model is invoked **once**.  The result is
        reshaped to ``(n_out·n_years, H*W)`` — out-band major, year minor —
        and appended to ``rdata.array``; one info row per (output, year) is
        added under ``group`` (default ``"prediction"``).  Each row carries a
        ``year`` column (the prediction year, or ``None`` for a static
        catalogue).

        When ``predict_proba=True`` a dominant-class band (the argmax over the
        probability classes) is appended in addition to the per-class
        probability bands, named ``prediction`` (one band per year).

        Years come from the temporal layers' ``start_date``; a static-only
        catalogue yields ``n_years == 1`` (a single prediction).

        :param model: a fitted model exposing ``predict`` (and optionally
          ``predict_proba``).
        :param feature_names: covariate names in the model's feature order.
          Defaults to ``model.feature_names_in_`` (or ``feature_names_``).
        :param predict_proba: use ``model.predict_proba`` instead of
          ``model.predict`` (default).
        :param valid_only: select which pixels to predict.

          * ``True`` (default) — per-(year, pixel) NaN validity.
          * ``False`` — predict every pixel for every year.
          * a 1-D boolean ``np.ndarray`` of shape ``(H*W,)`` — a static land
            mask, applied to **all** years.
        :param target_names: output names for multi-output / probability
          models.  Replaces the ``prob_0``/``out_0`` suffix with the given
          names (``prediction_<name>``).  Must have ``n_out`` entries.
        :param group: info group for the appended prediction bands.
        """

        def __init__(
            self,
            model,
            feature_names=None,
            predict_proba: bool = False,
            valid_only=True,
            target_names=None,
            group: str = "prediction",
            verbose: bool = True,
        ) -> None:
            super().__init__(verbose=verbose)
            self.model = model
            self.feature_names = feature_names
            self.predict_proba = predict_proba
            self.valid_only = valid_only
            self.target_names = target_names
            self.group = group

        def _resolve_feature_names(self):
            feature_names = self.feature_names
            if feature_names is None:
                feature_names = getattr(self.model, "feature_names_in_", None)
                if feature_names is None:
                    feature_names = getattr(self.model, "feature_names_", None)
                if feature_names is None:
                    raise ValueError(
                        "model has no feature_names_in_/feature_names_; "
                        "pass feature_names="
                    )
            return list(feature_names)

        def _out_names(self, n_out):
            if self.target_names is not None:
                if len(self.target_names) != n_out:
                    raise ValueError(
                        f"target_names has {len(self.target_names)} entries but "
                        f"the model produces {n_out} outputs"
                    )
                return [f"prediction_{t}" for t in self.target_names]
            if n_out == 1:
                return ["prediction"]
            if self.predict_proba:
                return [f"prediction_prob_{i}" for i in range(n_out)]
            return [f"prediction_out_{i}" for i in range(n_out)]

        def _predict(self, rdata):
            feature_names = self._resolve_feature_names()
            covs_idx, years = rdata._get_covs_idx_by_year(feature_names)
            arr = rdata.array.get()
            n_pixels = arr.shape[1]
            n_years = len(years)

            # Build the concatenated feature matrix (n_years·n_pixels,
            # n_features), rows ordered year-major.  Static covariates repeat
            # because their column in covs_idx is the same band every year.
            # Pre-allocate and fill in-place: the old np.concatenate over a
            # list of per-year transposed copies held n_years blocks at once.
            n_features = covs_idx.shape[0]
            X = np.empty((n_years * n_pixels, n_features), dtype=np.float32)
            for j in range(n_years):
                X[j * n_pixels : (j + 1) * n_pixels, :] = arr[
                    covs_idx[:, j], :
                ].T

            if isinstance(self.valid_only, np.ndarray):
                valid = np.tile(np.asarray(self.valid_only, dtype=bool), n_years)
            elif self.valid_only:
                valid = ~np.isnan(X).any(axis=1)
            else:
                valid = np.ones(X.shape[0], dtype=bool)

            fn = self.model.predict_proba if self.predict_proba else self.model.predict

            pred_full = np.full((n_years * n_pixels, 1), np.nan, dtype=np.float32)
            if valid.any():
                pred_valid = np.asarray(fn(X[valid]), dtype=np.float32)
                if pred_valid.ndim == 1:
                    pred_valid = pred_valid.reshape(-1, 1)
                n_out = pred_valid.shape[1]
                pred_full = np.full(
                    (n_years * n_pixels, n_out), np.nan, dtype=np.float32
                )
                pred_full[valid] = pred_valid
            else:
                n_out = pred_full.shape[1]

            # (n_years, n_pixels, n_out) -> (n_out, n_years, n_pixels) ->
            # (n_out·n_years, n_pixels), out-band major, year minor.
            pred = (
                pred_full.reshape(n_years, n_pixels, n_out)
                .transpose(2, 0, 1)
                .reshape(n_out * n_years, n_pixels)
            )
            # Dominant class (argmax over classes) for predict_proba.
            dominant = None
            if self.predict_proba and n_out > 1:
                proba_3d = pred.reshape(n_out, n_years, n_pixels)
                valid_yp = ~np.isnan(proba_3d).all(axis=0)
                dominant = np.full((n_years, n_pixels), np.nan, dtype=np.float32)
                if valid_yp.any():
                    dominant[valid_yp] = (
                        proba_3d[:, valid_yp].argmax(axis=0).astype(np.float32)
                    )

            return pred, dominant, years, n_out

        def _info_row(self, rdata, name, year):
            """Build one info row for a prediction band, setting the ``year``."""
            if year is None:
                row = rdata._new_info_row(
                    rdata.base_raster, group=self.group, name=name
                )
                row[RasterData.TEMPORAL_COL] = False
                row["year"] = None
                return row
            row = rdata._new_info_row(
                rdata.base_raster,
                group=self.group,
                name=name,
                dates=[f"{year}-01-01", f"{year}-12-31"],
                date_format="%Y-%m-%d",
                date_style="interval",
            )
            row["year"] = year
            return row

        def run(self, rdata, outname=None):
            pred, dominant, years, n_out = self._predict(rdata)
            out_names = self._out_names(n_out)

            arr = rdata.array.get()
            new_bands = [arr, pred]
            if dominant is not None:
                new_bands.append(dominant)

            rdata.array = parallel.put_shared(
                np.concatenate(new_bands, axis=0),
                local=rdata.backend.name == "cpp",
            )

            new_info = []
            # probability / label bands: out-band major, year minor
            for out in range(n_out):
                for year in years:
                    new_info.append(self._info_row(rdata, out_names[out], year))
            # dominant-class bands (argmax): one per year
            if dominant is not None:
                for year in years:
                    new_info.append(self._info_row(rdata, "prediction", year))
            return None, DataFrame(new_info)

    class WhaleRunner(SKMapRunner):
        """Base class for on-the-fly feature (whale) runners.

        Each whale computes one derived band from existing bands (looked up by
        name via :meth:`RasterData._band_index`) and appends it to
        ``rdata.array``. The new band is added to ``info`` under ``group``
        (default: the primary input band's group) and ``name`` (``outname`` or
        a class default).

        # ponytail: computed in the main process with numpy vectorization; a
        # whale touches a handful of bands so materializing is negligible.
        # Route through a worker only if a whale becomes a throughput hotspot.
        """

        def __init__(self, verbose: bool = True) -> None:
            super().__init__(verbose=verbose)

        def _band(self, rdata: RasterData, name: str, group: str = None) -> np.ndarray:
            return rdata.array.get()[rdata._band_index(name, group), :]

        def _primary_band(self) -> str:
            raise NotImplementedError

        def _compute(self, rdata: RasterData, arr: np.ndarray) -> np.ndarray:
            raise NotImplementedError

        def run(self, rdata: RasterData, outname: str = None, group: str = None):
            arr = rdata.array.get()
            new_band = np.asarray(self._compute(rdata, arr), dtype=arr.dtype)
            new_band = new_band.reshape(1, -1)

            if group is None:
                primary = rdata._band_index(self._primary_band())
                group = rdata.info.iloc[primary][RasterData.GROUP_COL]
            name = outname or self.__class__.__name__.lower()

            rdata.array = parallel.put_shared(
                np.concatenate([arr, new_band], axis=0),
                local=rdata.backend.name == "cpp",
            )
            new_info = DataFrame(
                [rdata._new_info_row(rdata.base_raster, group=group, name=name)]
            )
            return None, new_info

    class NormalizedDifference(WhaleRunner):
        """Derive a normalised-difference band from two input bands.

        ``val = (plus*scale_plus - minus*scale_minus) /
        (plus*scale_plus + minus*scale_minus) * scale_result + offset_result``
        (rounded, clipped; infinities mapped to ±scale_result + offset).
        """

        def __init__(
            self,
            idx_plus: str,
            idx_minus: str,
            scale_plus: float = 1.0,
            scale_minus: float = 1.0,
            scale_result: float = 1.0,
            offset_result: float = 0.0,
            clip: list = None,
            verbose: bool = True,
        ) -> None:
            super().__init__(verbose=verbose)
            self.idx_plus = idx_plus
            self.idx_minus = idx_minus
            self.scale_plus = scale_plus
            self.scale_minus = scale_minus
            self.scale_result = scale_result
            self.offset_result = offset_result
            self.clip = clip if clip is not None else [-np.inf, np.inf]
            if len(self.clip) == 1:
                self.clip = [-self.clip[0], self.clip[0]]

        def _primary_band(self):
            return self.idx_plus

        def _compute(self, rdata, arr):
            p = self._band(rdata, self.idx_plus) * self.scale_plus
            m = self._band(rdata, self.idx_minus) * self.scale_minus
            with np.errstate(divide="ignore", invalid="ignore"):
                val = (p - m) / (p + m) * self.scale_result + self.offset_result
            val = np.round(val)
            val = np.where(val == -np.inf, -self.scale_result + self.offset_result, val)
            val = np.where(val == np.inf, self.scale_result + self.offset_result, val)
            return np.clip(val, self.clip[0], self.clip[1])

    class Nirv(WhaleRunner):
        """Derive NIRv = ((NDVI - 0.08) * NIR) scaled."""

        def __init__(
            self,
            idx_nir: str,
            idx_red: str,
            nir_scaling: float = 1.0,
            red_scaling: float = 1.0,
            result_scaling: float = 1.0,
            result_offset: float = 0.0,
            clip: list = None,
            verbose: bool = True,
        ) -> None:
            super().__init__(verbose=verbose)
            self.idx_nir = idx_nir
            self.idx_red = idx_red
            self.nir_scaling = nir_scaling
            self.red_scaling = red_scaling
            self.result_scaling = result_scaling
            self.result_offset = result_offset
            self.clip = clip if clip is not None else [-np.inf, np.inf]
            if len(self.clip) == 1:
                self.clip = [-self.clip[0], self.clip[0]]

        def _primary_band(self):
            return self.idx_nir

        def _compute(self, rdata, arr):
            nir = self._band(rdata, self.idx_nir) * self.nir_scaling
            red = self._band(rdata, self.idx_red) * self.red_scaling
            with np.errstate(divide="ignore", invalid="ignore"):
                ndvi = (nir - red) / (nir + red)
                val = ((ndvi - 0.08) * nir) * self.result_scaling + self.result_offset
            val = np.round(val)
            val = np.where(val == -np.inf, -self.result_scaling + self.result_offset, val)
            val = np.where(val == np.inf, self.result_scaling + self.result_offset, val)
            return np.clip(val, self.clip[0], self.clip[1])

    class Savi(WhaleRunner):
        """Derive SAVI = (nir - red) * 1.5 / (nir + red + 0.5) scaled."""

        def __init__(
            self,
            idx_red: str,
            idx_nir: str,
            red_scaling: float = 1.0,
            nir_scaling: float = 1.0,
            result_scaling: float = 1.0,
            result_offset: float = 0.0,
            clip: list = None,
            verbose: bool = True,
        ) -> None:
            super().__init__(verbose=verbose)
            self.idx_red = idx_red
            self.idx_nir = idx_nir
            self.red_scaling = red_scaling
            self.nir_scaling = nir_scaling
            self.result_scaling = result_scaling
            self.result_offset = result_offset
            self.clip = clip if clip is not None else [-np.inf, np.inf]
            if len(self.clip) == 1:
                self.clip = [-self.clip[0], self.clip[0]]

        def _primary_band(self):
            return self.idx_nir

        def _compute(self, rdata, arr):
            nir = self._band(rdata, self.idx_nir) * self.nir_scaling
            red = self._band(rdata, self.idx_red) * self.red_scaling
            with np.errstate(divide="ignore", invalid="ignore"):
                val = ((nir - red) * 1.5) / (nir + red + 0.5)
                val = val * self.result_scaling + self.result_offset
            val = np.round(val)
            val = np.where(val == -np.inf, -self.result_scaling + self.result_offset, val)
            val = np.where(val == np.inf, self.result_scaling + self.result_offset, val)
            return np.clip(val, self.clip[0], self.clip[1])

    class ExtractIndicator(WhaleRunner):
        """Derive a binary indicator band: 1 where ``layer == code``, else 0."""

        def __init__(self, idx_layer: str, code: float, verbose: bool = True) -> None:
            super().__init__(verbose=verbose)
            self.idx_layer = idx_layer
            self.code = code

        def _primary_band(self):
            return self.idx_layer

        def _compute(self, rdata, arr):
            return (self._band(rdata, self.idx_layer) == self.code).astype(np.float32)

    class PercentileAggregation(WhaleRunner):
        """Derive a percentile band across a set of named temporal bands."""

        def __init__(self, bands: list, percentile: float, verbose: bool = True) -> None:
            super().__init__(verbose=verbose)
            self.bands = bands
            self.percentile = percentile

        def _primary_band(self):
            return self.bands[0]

        def _compute(self, rdata, arr):
            idxs = [rdata._band_index(b) for b in self.bands]
            return np.nanpercentile(arr[idxs, :], self.percentile, axis=0)

    class GetLatitude(WhaleRunner):
        """Derive a latitude band from the base raster's geotransform."""

        def _primary_band(self):
            raise RuntimeError("GetLatitude has no input band")

        def run(self, rdata: RasterData, outname: str = None, group: str = None):
            import rasterio

            arr = rdata.array.get()
            h, w = rdata._spatial_shape
            row_off = rdata.window.row_off if rdata.window is not None else 0
            with rasterio.open(rdata.base_raster) as src:
                transform = src.transform
            lats = transform.f + (np.arange(h) + row_off + 0.5) * transform.e
            new_band = np.repeat(lats, w).astype(arr.dtype).reshape(1, -1)

            if group is None:
                group = "common"
            name = outname or "latitude"

            rdata.array = parallel.put_shared(
                np.concatenate([arr, new_band], axis=0),
                local=rdata.backend.name == "cpp",
            )
            new_info = DataFrame(
                [rdata._new_info_row(rdata.base_raster, group=group, name=name)]
            )
            return None, new_info

    class GeometricTemperature(WhaleRunner):
        """Derive a geometric-temperature band from latitude and elevation."""

        def __init__(
            self,
            idx_latitude: str,
            idx_elevation: str,
            elevation_scaling: float = 1.0,
            a: float = 1.0,
            b: float = 1.0,
            result_scaling: float = 1.0,
            day_of_year_mmdd: str = "0101",
            verbose: bool = True,
        ) -> None:
            super().__init__(verbose=verbose)
            self.idx_latitude = idx_latitude
            self.idx_elevation = idx_elevation
            self.elevation_scaling = elevation_scaling
            self.a = a
            self.b = b
            self.result_scaling = result_scaling
            from skmap.misc import mmdd_to_doy

            self.day_of_year = mmdd_to_doy(day_of_year_mmdd)

        def _primary_band(self):
            return self.idx_elevation

        def _compute(self, rdata, arr):
            lat = self._band(rdata, self.idx_latitude)
            elev = self._band(rdata, self.idx_elevation)
            doy = self.day_of_year
            cos_teta = np.cos(((doy - 18.0) / 182.5 + 4.0 ** (lat < 0)) * np.pi)
            cos_fi = np.cos(lat * np.pi / 180.0)
            sin_abs_fi = np.abs(np.sin(lat * np.pi / 180.0))
            res = self.a * cos_fi + self.b * (1.0 - cos_teta) * sin_abs_fi
            res = res - 0.006 * self.elevation_scaling * elev
            return np.round(res * self.result_scaling)

except ImportError as e:
    from skmap.misc import _warn_deps

    _warn_deps(e, "skmap.io")

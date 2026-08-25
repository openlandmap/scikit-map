from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

__all__: Sequence[str] = [
    "applyTsirf",
    "averageAggregate",
    "blocksAverage",
    "blocksAverageVecs",
    "castFloat32ToFloat64",
    "castFloat64ToFloat32",
    "checkSimdInstructionSetsInUse",
    "computeBsi",
    "computeEvi",
    "computeFapar",
    "computeGeometricTemperature",
    "computeMannKendallPValues",
    "computeNirv",
    "computeNormalizedDifference",
    "computePercentiles",
    "computeSavi",
    "convolveRows",
    "copyVecInMatrixRow",
    "elementwiseAverage",
    "expandArrayCols",
    "expandArrayRows",
    "extractIndicators",
    "extractOverlay",
    "fillArray",
    "fitPercentage",
    "fitProbabilities",
    "getLatLonArray",
    "hadamardProduct",
    "inverseReorderArray",
    "linearRegression",
    "maskData",
    "maskDataRows",
    "maskDifference",
    "maskNan",
    "maskNanRows",
    "nanMean",
    "nanMeanAggregatePattern",
    "offsetAndScale",
    "offsetsAndScales",
    "readData",
    "readDataBlocks",
    "readDataCore",
    "reorderArray",
    "scaleAndOffset",
    "selArrayCols",
    "selArrayRows",
    "slidingWindowClassMode",
    "swapRowsValues",
    "texturesBwTransform",
    "transposeArray",
    "transposeReorderArray",
    "writeData",
]

def applyTsirf(
    data: NDArray[np.float32],
    n_threads: int,
    out_data: NDArray[np.float32],
    out_index_offset: int,
    w_0: float,
    w_p: NDArray[np.float32],
    w_f: NDArray[np.float32],
    keep_original_values: bool,
    version: str,
    backend: str,
) -> None:
    """
    Apply TSIRF
    """

def averageAggregate(
    data: NDArray[np.float32],
    n_threads: int,
    out_data: NDArray[np.float32],
    agg_factor: int,
) -> None:
    """
    Average aggregate
    """

def blocksAverage(
    out: NDArray[np.float32],
    n_threads: int,
    in1: NDArray[np.float32],
    in2: NDArray[np.float32],
    n_pix: int,
    y: int,
) -> None:
    """
    Vecorized average of 4 neighbor elemnts
    """

def blocksAverageVecs(
    out: NDArray[np.float32],
    n_threads: int,
    in1: NDArray[np.float32],
    in2: NDArray[np.float32],
    n_pix: int,
    y: int,
    row_offsets: int,
) -> None:
    """
    Vecorized average of 4 neighbor elemnts
    """

def castFloat32ToFloat64(
    data: NDArray[np.float32],
    n_threads: int,
    out_data: NDArray[np.float32],
) -> None: ...
def castFloat64ToFloat32(
    data: NDArray[np.float32],
    n_threads: int,
    out_data: NDArray[np.float32],
) -> None: ...
def checkSimdInstructionSetsInUse() -> None: ...
def computeBsi(
    data: NDArray[np.float32],
    n_threads: int,
    swir1_indices: Sequence[int],
    red_indices: Sequence[int],
    nir_indices: Sequence[int],
    blue_indices: Sequence[int],
    result_indices: Sequence[int],
    swir1_scaling: float,
    red_scaling: float,
    nir_scaling: float,
    blue_scaling: float,
    result_scaling: float,
    result_offset: float,
    clip_value: Sequence[float],
) -> None:
    """
    Compute BSI
    """

def computeEvi(
    data: NDArray[np.float32],
    n_threads: int,
    red_indices: Sequence[int],
    nir_indices: Sequence[int],
    blue_indices: Sequence[int],
    result_indices: Sequence[int],
    red_scaling: float,
    nir_scaling: float,
    blue_scaling: float,
    result_scaling: float,
    result_offset: float,
    clip_value: Sequence[float],
) -> None: ...
def computeFapar(
    data: NDArray[np.float32],
    n_threads: int,
    red_indices: Sequence[int],
    nir_indices: Sequence[int],
    result_indices: Sequence[int],
    red_scaling: float,
    nir_scaling: float,
    result_scaling: float,
    result_offset: float,
    clip_value: Sequence[float],
) -> None: ...
def computeGeometricTemperature(
    data: NDArray[np.float32],
    n_threads: int,
    latitude: NDArray[np.float32],
    elevation: NDArray[np.float32],
    elevation_scaling: float,
    a: float,
    b: float,
    result_scaling: float,
    result_indices: Sequence[int],
    days_of_year: Sequence[float],
) -> None: ...
def computeMannKendallPValues(
    data: NDArray[np.float32],
    n_threads: int,
    out_data: NDArray[np.float32],
) -> None: ...
def computeNirv(
    data: NDArray[np.float32],
    n_threads: int,
    nir_indices: Sequence[int],
    red_indices: Sequence[int],
    result_indices: Sequence[int],
    nir_scaling: float,
    red_scaling: float,
    result_scaling: float,
    result_offset: float,
    clip_value: Sequence[float],
) -> None: ...
def computeNormalizedDifference(
    data: NDArray[np.float32],
    n_threads: int,
    positive_indices: Sequence[int],
    negative_indices: Sequence[int],
    result_indices: Sequence[int],
    positive_scaling: float,
    negative_scaling: float,
    result_scaling: float,
    result_offset: float,
    clip_value: Sequence[float],
) -> None: ...
def computePercentiles(
    data: NDArray[np.float32],
    n_threads: int,
    col_in_select: Sequence[int],
    out_data: NDArray[np.float32],
    col_out_select: Sequence[int],
    percentiles: Sequence[float],
) -> None: ...
def computeSavi(
    data: NDArray[np.float32],
    n_threads: int,
    red_indices: Sequence[int],
    nir_indices: Sequence[int],
    result_indices: Sequence[int],
    red_scaling: float,
    nir_scaling: float,
    result_scaling: float,
    result_offset: float,
    clip_value: Sequence[float],
) -> None: ...
def convolveRows(
    data: NDArray[np.float32],
    n_threads: int,
    out_data: NDArray[np.float32],
    w_0: float,
    w_p: NDArray[np.float32],
    w_f: NDArray[np.float32],
) -> None: ...
def copyVecInMatrixRow(
    data: NDArray[np.float32],
    n_threads: int,
    in_vec: NDArray[np.float32],
    row_idx: int,
) -> None: ...
def elementwiseAverage(
    out: NDArray[np.float32],
    n_threads: int,
    in1: NDArray[np.float32],
    in2: NDArray[np.float32],
) -> None: ...
def expandArrayCols(
    data: NDArray[np.float32],
    n_threads: int,
    out_data: NDArray[np.float32],
    col_select: Sequence[int],
) -> None: ...
def expandArrayRows(
    data: NDArray[np.float32],
    n_threads: int,
    out_data: NDArray[np.float32],
    row_select: Sequence[int],
) -> None: ...
def extractIndicators(
    data_in: NDArray[np.float32],
    n_threads: int,
    data_out: NDArray[np.float32],
    col_in_select: int,
    col_out_select: Sequence[int],
    classes: Sequence[int],
) -> None: ...
def extractOverlay(
    data: NDArray[np.float32],
    n_threads: int,
    pix_block_ids: Sequence[int],
    pix_inblock_idxs: Sequence[int],
    unique_blocks_ids_comb: Sequence[int],
    key_layer_ids_comb: Sequence[int],
    data_overlay: NDArray[np.float32],
) -> None: ...
def fillArray(
    data: NDArray[np.float32],
    n_threads: int,
    val: float,
) -> None: ...
def fitPercentage(
    out: NDArray[np.float32],
    n_threads: int,
    in1: NDArray[np.float32],
    in2: NDArray[np.float32],
) -> None: ...
def fitProbabilities(
    data: NDArray[np.float32],
    n_threads: int,
    out_data: NDArray[np.float32],
    input_scaling: float,
    target_scaling: int,
    best_classes_data: NDArray[np.float32],
    n_best_classes: int,
) -> None: ...
def getLatLonArray(
    data: NDArray[np.float32],
    n_threads: int,
    conf_GDAL: dict,
    file_loc: str | Path,
    x_off: int,
    y_off: int,
    x_size: int,
    y_size: int,
) -> None: ...
def hadamardProduct(
    out: NDArray[np.float32],
    n_threads: int,
    in1: NDArray[np.float32],
    in2: NDArray[np.float32],
) -> None: ...
def inverseReorderArray(
    data: NDArray[np.float32],
    n_threads: int,
    out_data: NDArray[np.float32],
    indices_matrix: Sequence[Sequence[int]],
) -> None: ...
def linearRegression(
    data: NDArray[np.float32],
    n_threads: int,
    x: NDArray[np.float32],
    beta_0: NDArray[np.float32],
    beta_1: NDArray[np.float32],
) -> None: ...
def maskData(
    data: NDArray[np.float32],
    n_threads: int,
    row_select: Sequence[int],
    mask: np.ndarray[Any, np.dtype[np.float32]],
    value_of_mask_to_mask: float,
    new_value_in_data: float,
) -> None: ...
def maskDataRows(
    data: NDArray[np.float32],
    n_threads: int,
    row_select: Sequence[int],
    mask: NDArray[np.float32],
    value_of_mask_to_mask: float,
    new_value_in_data: float,
) -> None: ...
def maskDifference(
    data: NDArray[np.float32],
    n_threads: int,
    diff_th: float,
    count_th: int,
    ref_data: NDArray[np.float32],
    mask_out: NDArray[np.float32],
) -> None: ...
def maskNan(
    data: NDArray[np.float32],
    n_threads: int,
    row_select: Sequence[int],
    new_value_in_data: float,
) -> None: ...
def maskNanRows(
    data: NDArray[np.float32],
    n_threads: int,
    row_select: Sequence[int],
    new_value_vec: NDArray[np.float32],
) -> None: ...
def nanMean(
    data: NDArray[np.float32],
    n_threads: int,
    out_data: NDArray[np.float32],
) -> None: ...
def nanMeanAggregatePattern(
    data: NDArray[np.float32],
    n_threads: int,
    out_data: NDArray[np.float32],
    agg_pattern: Sequence[Sequence[int]],
) -> None: ...
def offsetAndScale(
    data: NDArray[np.float32],
    n_threads: int,
    offset: float,
    scaling: float,
) -> None: ...
def offsetsAndScales(
    data: NDArray[np.float32],
    n_threads: int,
    row_select: Sequence[int],
    offsets: NDArray[np.float32],
    scalings: NDArray[np.float32],
) -> None: ...
def readData(
    data: NDArray[np.float32],
    n_threads: int,
    file_locs: Sequence[str],
    perm_vec: Sequence[int],
    x_off: int,
    y_off: int,
    x_size: int,
    y_size: int,
    bands_listr: Sequence[int],
    conf_GDAL: dict,
    value_to_mask: float | None = None,
    value_to_set: float | None = None,
    overview: int = 0,
) -> None: ...
def readDataBlocks(
    data: NDArray[np.float32],
    n_threads: int,
    file_locs: Sequence[str],
    perm_vec: Sequence[int],
    x_off_vec: Sequence[int],
    y_off_vec: Sequence[int],
    x_size_vec: Sequence[int],
    y_size_vec: Sequence[int],
    bands_list: Sequence[int],
    conf_GDAL: dict,
    value_to_mask_vec: Sequence[float] | None = None,
    value_to_set: float | None = None,
) -> None: ...
def readDataCore(
    data: NDArray[np.float32],
    n_threads: int,
    file_loc: str,
    x_off: int,
    y_off: int,
    x_size: int,
    y_size: int,
    bands_list: Sequence[int],
    conf_GDAL: dict,
    value_to_mask: float | None = None,
    value_to_set: float | None = None,
    overview: int = 0,
) -> None: ...
def reorderArray(
    data: NDArray[np.float32],
    n_threads: int,
    out_data: NDArray[np.float32],
    indices_matrix: Sequence[Sequence[int]],
) -> None: ...
def scaleAndOffset(
    data: NDArray[np.float32],
    n_threads: int,
    offset: float,
    scaling: float,
) -> None: ...
def selArrayCols(
    data: NDArray[np.float32],
    n_threads: int,
    out_data: NDArray[np.float32],
    col_select: Sequence[int],
) -> None: ...
def selArrayRows(
    data: NDArray[np.float32],
    n_threads: int,
    out_data: NDArray[np.float32],
    row_select: Sequence[int],
) -> None: ...
def slidingWindowClassMode(
    data: NDArray[np.float32],
    n_threads: int,
    out_data: NDArray[np.float32],
    window_size: int,
) -> None: ...
def swapRowsValues(
    data: NDArray[np.float32],
    n_threads: int,
    row_select: Sequence[int],
    value_to_mask: float,
    new_value: float,
) -> None: ...
def texturesBwTransform(
    texture_1: NDArray[np.float32],
    n_threads: int,
    texture_2: NDArray[np.float32],
    k: float,
    a: float,
    sand: NDArray[np.float32],
    silt: NDArray[np.float32],
    clay: NDArray[np.float32],
) -> None: ...
def transposeArray(
    data: NDArray[np.float32],
    n_threads: int,
    out_data: NDArray[np.float32],
) -> None: ...
def transposeReorderArray(
    data: NDArray[np.float32],
    n_threads: int,
    out_data: NDArray[np.float32],
    permutation_matrix: Sequence[Sequence[int]],
) -> None: ...
def writeData(
    data: NDArray[np.float32],
    n_threads: int,
    conf_GDAL: dict,
    base_files: Sequence[str],
    base_folder: str,
    file_names: Sequence[str],
    data_indices: Sequence[int],
    x_off: int,
    y_off: int,
    x_size: int,
    y_size: int,
    no_data_value: float,
    gdal_data_type_str: str,
    creation_options: Sequence[str] = (),
    scale: float = 1.0,
) -> None: ...

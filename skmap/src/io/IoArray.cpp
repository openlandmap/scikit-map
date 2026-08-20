#include "io/IoArray.h"

#include <fstream>
#include <mutex>

namespace skmap {

// Bug fix: GDALAllRegister() must only be called once. Calling it on every
// Python invocation stresses GDAL's driver-registry mutex and accumulates
// internal state across loop iterations. The error-handler setup is also
// global state (CPLPushErrorHandler grows a stack on every call), so it is
// gated behind the same once_flag.
static std::once_flag s_gdal_init_flag;
static void initGdal() {
  GDALAllRegister();
  std::ofstream nullStream("/dev/null");
  if (nullStream.is_open()) {
    CPLSetErrorHandler(CPLLoggingErrorHandler);
    CPLSetConfigOption("CPL_LOG", "/dev/null");
  } else {
    CPLPushErrorHandler(CPLQuietErrorHandler);
  }
}

GDALResampleAlg hashResample(std::string const &inString) {
  if (inString == "GRA_CubicSpline")
    return GRA_CubicSpline;
  if (inString == "GRA_NearestNeighbour")
    return GRA_NearestNeighbour;
  if (inString == "GRA_Bilinear")
    return GRA_Bilinear;
  if (inString == "GRA_Cubic")
    return GRA_Cubic;
  if (inString == "GRA_Lanczos")
    return GRA_Lanczos;
  if (inString == "GRA_Average")
    return GRA_Average;
  if (inString == "GRA_Mode")
    return GRA_Mode;
  if (inString == "GRA_Max")
    return GRA_Max;
  if (inString == "GRA_Min")
    return GRA_Min;
  if (inString == "GRA_Med")
    return GRA_Med;
  if (inString == "GRA_Q1")
    return GRA_Q1;
  if (inString == "GRA_Q3")
    return GRA_Q3;
  if (inString == "GRA_Sum")
    return GRA_Sum;
  if (inString == "GRA_RMS")
    return GRA_RMS;
  skmapAssertIfTrue(
      true,
      "scikit-map ERROR 38: failed to read raster data into Eigen matrix");
  return GRA_NearestNeighbour;
}

void IoArray::warpTile(std::string ref_tile_path, std::string mosaic_path,
                       std::string resample) {

  // @FIXME: this function assumes that the data is single band
  // Extracting reference metadata
  GdalDatasetGuard refTileDataset(
      (GDALDataset *)GDALOpen(ref_tile_path.c_str(), GA_ReadOnly));
  skmapAssertIfTrue(
      refTileDataset.get() == nullptr,
      "scikit-map ERROR 24: issues in opening ref_tile_path with path " +
          ref_tile_path);

  double ref_geotransform[6];
  double min_x, max_y, pixel_width, pixel_height, max_x, min_y;
  refTileDataset->GetGeoTransform(ref_geotransform);
  min_x = ref_geotransform[0];
  max_y = ref_geotransform[3];
  pixel_width = ref_geotransform[1];
  pixel_height = ref_geotransform[5];

  // Determine the size of the output raster
  // Forcing rounding because rounding strategy for float to int is always
  // truncation in C++
  max_x = min_x + (refTileDataset->GetRasterXSize() * pixel_width);
  min_y = max_y + (refTileDataset->GetRasterYSize() * pixel_height);
  double target_geotransform[6] = {min_x, pixel_width, 0,
                                   max_y, 0,           pixel_height};
  uint_t target_x_size =
      (uint_t)std::abs(std::round(((max_x - min_x) / pixel_width)));
  uint_t target_y_size =
      (uint_t)std::abs(std::round(((max_y - min_y) / pixel_height)));

  //            skmapAssertIfTrue((uint_t) row.cols() != target_x_size *
  //            target_y_size,
  //                              "scikit-map ERROR 26: array data columns are "
  //                              + std::to_string(row.cols()) + " while
  //                              target_x_size is " +
  //                              std::to_string(target_x_size) + " and
  //                              target_y_size is " +
  //                              std::to_string(target_y_size) );
  const char *projectionRef = refTileDataset->GetProjectionRef();
  skmapAssertIfTrue(projectionRef == nullptr,
                    "scikit-map ERROR 27: failed to get the projection system");
  OGRSpatialReference oSRS;
  auto ret_osrs_import = oSRS.importFromWkt(projectionRef);
  skmapAssertIfTrue(
      ret_osrs_import != OGRERR_NONE,
      "scikit-map ERROR 28: to import projection system from WKT");
  char *pszSRS_WKT_raw = nullptr;
  auto ret_osrs_export = oSRS.exportToWkt(&pszSRS_WKT_raw);
  skmapAssertIfTrue(ret_osrs_export != OGRERR_NONE,
                    "scikit-map ERROR 29: to export projection system to WKT");
  CplMemGuard pszSRS_WKT(pszSRS_WKT_raw);

  // Setting source warp options
  GdalDatasetGuard mosaicDataset(
      (GDALDataset *)GDALOpen(mosaic_path.c_str(), GA_ReadOnly));
  skmapAssertIfTrue(
      mosaicDataset.get() == nullptr,
      "scikit-map ERROR 30: issues in opening mosaic_path with path " +
          mosaic_path);

  // Retrieve NoData value from the mosaic
  GDALRasterBandH band = GDALGetRasterBand(mosaicDataset.get(), 1);
  int bSuccess = FALSE;
  double nodata_val = GDALGetRasterNoDataValue(band, &bSuccess);

  GdalWarpOptionsGuard psWarpOptions(GDALCreateWarpOptions());
  psWarpOptions.get()->hSrcDS = mosaicDataset.get();
  psWarpOptions.get()->nBandCount = 1;
  // @FIXME: check if this works in general
  psWarpOptions.get()->panSrcBands = (int *)CPLMalloc(sizeof(int));
  psWarpOptions.get()->panDstBands = (int *)CPLMalloc(sizeof(int));
  psWarpOptions.get()->panSrcBands[0] = 1;
  psWarpOptions.get()->panDstBands[0] = 1;

  // Set NoData value in warp options
  if (bSuccess) {
    GDALSetRasterNoDataValue(band, nodata_val);
    psWarpOptions.get()->padfSrcNoDataReal = (double *)CPLMalloc(sizeof(double));
    psWarpOptions.get()->padfSrcNoDataReal[0] = nodata_val;
  }

  // Setting target warp options
  // @FIXME: this currently work only for float32, specialize the function with
  // a template based on the type of float_t
  GdalDatasetGuard dstDataset(
      GetGDALDriverManager()->GetDriverByName("MEM")->Create(
          "", target_x_size, target_y_size, 1, GDT_Float32, nullptr));
  dstDataset->SetGeoTransform(target_geotransform);
  dstDataset->SetProjection((const char *)pszSRS_WKT.get());
  psWarpOptions.get()->hDstDS = dstDataset.get();
  psWarpOptions.get()->pTransformerArg = GDALCreateGenImgProjTransformer(
      mosaicDataset.get(), mosaicDataset->GetProjectionRef(), dstDataset.get(),
      (const char *)pszSRS_WKT.get(), FALSE, 0.0, 1);
  psWarpOptions.get()->pfnTransformer = GDALGenImgProjTransform;
  psWarpOptions.get()->eResampleAlg = hashResample(resample);

  GDALRasterBand *poBand = dstDataset->GetRasterBand(1);
  GDALWarpOperation operation;
  operation.Initialize(psWarpOptions.get());
  operation.ChunkAndWarpImage(0, 0, dstDataset->GetRasterXSize(),
                              dstDataset->GetRasterYSize());
  CPLErr outRead = poBand->RasterIO(GF_Read, 0, 0, target_x_size, target_y_size,
                                    m_data.data(), target_x_size, target_y_size,
                                    GDT_Float32, 0, 0);
  skmapAssertIfTrue(
      outRead != CE_None,
      "scikit-map ERROR 31: failed to read raster data into Eigen matrix");

  // Mask NoData values
  if (bSuccess) {
    m_data = (m_data.array() == static_cast<float_t>(nodata_val))
                 .select(nan_v, m_data);
  }

  // Cleanup is handled by the RAII guards above (GdalDatasetGuard,
  // GdalWarpOptionsGuard, CplMemGuard) on every path, including throws.
}

IoArray::IoArray(Eigen::Ref<MatFloat> data, const uint_t n_threads)
    : ParArray(data, n_threads) {}

void IoArray::setupGdal(dict_t dict) {
  for (auto &pair : dict) {
    CPLSetConfigOption(pair.first.c_str(), pair.second.c_str());
  }
  std::call_once(s_gdal_init_flag, initGdal); // Bug fix: initialise only once
}

// Bug fix: signature changed from Eigen::Ref<MatFloat::RowXpr> to
// (float_t* row_ptr, uint_t row_n_elems).
//
// m_data is stored as Eigen::Ref<MatFloat>, so m_data.row(i) returns
// Block<Ref<MatFloat>,1,Dyn,true> — a different type from MatFloat::RowXpr
// (= Block<MatFloat,1,Dyn,true>). Eigen's Ref constructor silently evaluates
// a temporary copy when these types don't match exactly, so GDAL was writing
// into a temporary that was discarded when the lambda returned, causing heap
// corruption and a delayed SIGSEGV on subsequent loop iterations.
// Using a raw pointer guarantees GDAL always writes directly into m_data's
// backing buffer. Post-read masking uses Eigen::Map over the same pointer.
void IoArray::readDataCore(float_t *row_ptr, uint_t row_n_elems,
                           std::string file_loc, uint_t x_off, uint_t y_off,
                           uint_t x_size, uint_t y_size, GDALDataType read_type,
                           std::vector<int> bands_list,
                           std::optional<float_t> value_to_mask,
                           std::optional<float_t> value_to_set) {
  // Bug fix: GDAL writes x_size*y_size*n_bands floats with default band
  // spacing (nBandSpace=0). The old guard only checked x_size*y_size, missing
  // the band multiplier and silently overflowing the buffer for multi-band reads.
  skmapAssertIfTrue(
      row_n_elems < x_size * y_size * (uint_t)bands_list.size(),
      "scikit-map ERROR 1B: row buffer too small for the requested bands "
      "(need " + std::to_string(x_size * y_size * bands_list.size()) +
      " elements, got " + std::to_string(row_n_elems) + ")");

  GdalDatasetGuard readDataset(
      (GDALDataset *)GDALOpen(file_loc.c_str(), GA_ReadOnly));
  skmapAssertIfTrue(
      readDataset.get() == nullptr,
      "scikit-map ERROR 1: issues in opening the file with path " + file_loc);

  if (!value_to_mask.has_value() && value_to_set.has_value()) {
    int bSuccess = FALSE;
    GDALRasterBandH band = GDALGetRasterBand(readDataset.get(), 1);
    const double nodata_val = GDALGetRasterNoDataValue(band, &bSuccess);
    if (bSuccess == TRUE)
      value_to_mask = static_cast<float_t>(nodata_val);
  }

  CPLErr outRead = readDataset->RasterIO(
      GF_Read, x_off, y_off, x_size, y_size, row_ptr, x_size, y_size,
      read_type, static_cast<int>(bands_list.size()), bands_list.data(),
      0, 0, 0);
  skmapAssertIfTrue(outRead != CE_None,
                    "scikit-map ERROR 2: issues in reading the file with URL " +
                        file_loc);

  if (value_to_mask.has_value() && value_to_set.has_value() &&
      value_to_mask.value() != value_to_set.value()) {
    Eigen::Map<Eigen::Matrix<float_t, 1, Eigen::Dynamic, Eigen::RowMajor>>
        row_map(row_ptr, row_n_elems);
    row_map = (row_map.array() == value_to_mask.value())
                  .select(value_to_set.value(), row_map);
  }
}

void IoArray::readData(std::vector<std::string> file_locs,
                       std::vector<uint_t> perm_vec, uint_t x_off, uint_t y_off,
                       uint_t x_size, uint_t y_size, GDALDataType read_type,
                       std::vector<int> bands_list,
                       std::optional<float_t> value_to_mask,
                       std::optional<float_t> value_to_set) {
  // Bug fix: guard updated to include the band multiplier.
  skmapAssertIfTrue(
      (uint_t)m_data.cols() < x_size * y_size * (uint_t)bands_list.size(),
      "scikit-map ERROR 0A: row buffer smaller than x_size * y_size * n_bands");
  // Bug fix: lambda uses auto&& and passes row.data()/row.size() so that
  // readDataCore always writes directly into m_data's buffer (no Eigen copy).
  auto readTiff = [&](uint_t i, auto &&row) {
    std::string file_loc = file_locs[i];
    this->readDataCore(row.data(), static_cast<uint_t>(row.size()), file_loc,
                       x_off, y_off, x_size, y_size, read_type, bands_list,
                       value_to_mask, value_to_set);
  };
  this->parRowPerm(readTiff, perm_vec);
}

void IoArray::readDataBlocks(
    std::vector<std::string> file_locs, std::vector<uint_t> perm_vec,
    std::vector<uint_t> x_off_vec, std::vector<uint_t> y_off_vec,
    std::vector<uint_t> x_size_vec, std::vector<uint_t> y_size_vec,
    GDALDataType read_type, std::vector<int> bands_list,
    std::optional<std::vector<float_t>> value_to_mask_vec,
    std::optional<float_t> value_to_set) {
  // Bug fix: guard updated to include the band multiplier.
  skmapAssertIfTrue(
      (uint_t)m_data.cols() <
          (*std::max_element(x_size_vec.begin(), x_size_vec.end())) *
              (*std::max_element(y_size_vec.begin(), y_size_vec.end())) *
              (uint_t)bands_list.size(),
      "scikit-map ERROR 0B: row buffer smaller than max(x_size)*max(y_size)*n_bands");
  // Bug fix: same auto&& / raw-pointer approach as readData.
  auto readTiffBlock = [&](uint_t i, auto &&row) {
    std::string file_loc = file_locs[i];
    std::optional<float_t> value_to_mask_i =
        (value_to_mask_vec.has_value() && value_to_mask_vec->size() > i)
            ? std::optional<float_t>(value_to_mask_vec.value()[i])
            : std::nullopt;
    this->readDataCore(row.data(), static_cast<uint_t>(row.size()), file_loc,
                       x_off_vec[i], y_off_vec[i], x_size_vec[i], y_size_vec[i],
                       read_type, bands_list, value_to_mask_i, value_to_set);
  };
  this->parRowPerm(readTiffBlock, perm_vec);
}

void IoArray::getLatLonArray(std::string file_loc, uint_t x_off, uint_t y_off,
                             uint_t x_size, uint_t y_size) {
  GdalDatasetGuard readDataset(
      (GDALDataset *)GDALOpen(file_loc.c_str(), GA_ReadOnly));
  skmapAssertIfTrue(readDataset.get() == nullptr,
                    "scikit-map ERROR 6: issues in opening the file with URL " +
                        file_loc);
  skmapAssertIfTrue(((uint_t)m_data.cols() != x_size * y_size),
                    "scikit-map ERROR 7: size of the longitude-latitude array "
                    "should match the total number of pixels");

  double geotransform[6];
  readDataset->GetGeoTransform(geotransform);

  auto getLatLonArrayRow = [&](uint_t i) {
    for (uint_t j = 0; j < x_size; j++) {
      double x = geotransform[0] + (x_off + j) * geotransform[1] +
                 (y_off + i) * geotransform[2];
      double y = geotransform[3] + (x_off + j) * geotransform[4] +
                 (y_off + i) * geotransform[5];
      m_data(0, i * x_size + j) = (float_t)x; // Longitude
      m_data(1, i * y_size + j) = (float_t)y; // Latitude
    }
  };
  this->parForRange(getLatLonArrayRow, y_size);
}

void IoArray::extractOverlay(std::vector<uint_t> pix_block_ids,
                             std::vector<uint_t> pix_inblock_idxs,
                             std::vector<uint_t> unique_blocks_ids_comb,
                             std::vector<uint_t> key_layer_ids_comb,
                             Eigen::Ref<MatFloat> data_overlay) {
  uint_t n_pix = pix_block_ids.size();
  uint_t n_bids = unique_blocks_ids_comb.size();
  auto extractOverlayPix = [&](uint_t i) {
    uint bid = pix_block_ids[i];
    uint pid = pix_inblock_idxs[i];
    for (uint_t j = 0; j < n_bids; j++) {
      if (unique_blocks_ids_comb[j] == bid)
        data_overlay(key_layer_ids_comb[j], i) = m_data(j, pid);
    }
  };
  this->parForRange(extractOverlayPix, n_pix);
}

} // namespace skmap
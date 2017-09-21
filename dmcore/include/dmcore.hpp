#pragma once

#include <cstdint>
#include <tuple>

#include <mlpack/core.hpp>

using std::size_t;

using arma::mat;
using arma::Mat;
using arma::sp_mat;

#define DMCORE_VERSION_MAJOR 1
#define DMCORE_VERSION_MINOR 0
#define DMCORE_VERSION_PATCH 0

namespace dmcore {

enum class DistanceMetric {
	Euclidean,
	Cosine,
};

std::tuple<Mat<size_t>, mat> get_nn(mat data, size_t k, DistanceMetric metric=DistanceMetric::Euclidean);

sp_mat nn_to_mat(Mat<size_t> neighbors, mat distances);

}

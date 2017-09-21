#pragma once

#include <cstdint>
#include <tuple>

#include <mlpack/core.hpp>

using std::size_t;

using arma::mat;
using arma::Mat;

#define DMCORE_VERSION_MAJOR 1
#define DMCORE_VERSION_MINOR 0
#define DMCORE_VERSION_PATCH 0

namespace dmcore {

enum class DistanceMetric {
	Euclidean,
	Cosine,
};

std::tuple<Mat<size_t>, mat> get_nn(mat data, size_t k, DistanceMetric metric=DistanceMetric::Euclidean);

}

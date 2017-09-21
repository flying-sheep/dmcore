#pragma once

#include <cstdint>
#include <utility>

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

/**
 * Find the kNN in the data using a specific distance metric.
 */
std::pair<Mat<size_t>, mat> get_nn(mat data, size_t k, DistanceMetric metric=DistanceMetric::Euclidean);

/**
 * Create a sparse n×n distance matrix given two k×n matrices with neighbor indices and distances, respectively.
 */
sp_mat nn_to_mat(Mat<size_t> neighbors, mat distances);
sp_mat nn_to_mat(Mat<size_t> neighbors, mat distances, size_t k);

}

#pragma once

#include <cstdint>
#include <utility>

#include <mlpack/core.hpp>

using std::size_t;

using arma::mat;
using arma::Mat;
using arma::sp_mat;
using arma::vec;

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
std::pair<Mat<size_t>, mat> get_nn(const mat data, const size_t k, const DistanceMetric metric=DistanceMetric::Euclidean);

/**
 * Create a sparse n×n distance matrix given two k×n matrices with neighbor indices and distances, respectively.
 */
sp_mat nn_to_mat(const Mat<size_t> neighbors, const mat distances);
sp_mat nn_to_mat(const Mat<size_t> neighbors, const mat distances, size_t k);

sp_mat transition_probabilities(const sp_mat dists);
sp_mat transition_probabilities(const sp_mat dists, const double sigma); // global sigma
sp_mat transition_probabilities(const sp_mat dists, const vec sigmas);    // local sigmas

vec local_sigmas(const sp_mat dists, const size_t n_local);
vec local_sigmas(const sp_mat dists, const size_t n_local_start, const size_t n_local_end);

sp_mat get_norm_p(const sp_mat trans_p, const vec d);
sp_mat get_norm_p(const sp_mat trans_p, const vec d, const vec d_new);

}

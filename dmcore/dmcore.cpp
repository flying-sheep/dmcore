#include "dmcore.hpp"

#include <cstdint>
#include <utility>

#include <mlpack/core.hpp>
#include <mlpack/core/tree/cover_tree/typedef.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <armadillo>

using namespace dmcore;

using std::size_t;

using arma::mat;
using arma::Mat;
using arma::sp_mat;

using mlpack::neighbor::NeighborSearch;
using mlpack::neighbor::NearestNeighborSort;
using mlpack::metric::EuclideanDistance;
using mlpack::kernel::CosineDistance;
using mlpack::tree::StandardCoverTree;


template<typename MetricType>
std::pair<Mat<size_t>, mat>
get_nn_impl(mat data, size_t k) {
	NeighborSearch<NearestNeighborSort, MetricType, mat, StandardCoverTree> nn(data);
	
	Mat<size_t> neighbors;
	mat distances;
	nn.Search(k, neighbors, distances);
	
	return std::make_pair(neighbors, distances);
}

std::pair<Mat<size_t>, mat>
dmcore::get_nn(mat data, size_t k, DistanceMetric metric) {
	switch(metric) {
		case DistanceMetric::Euclidean: return get_nn_impl<EuclideanDistance>(data, k);
		case DistanceMetric::Cosine:    return get_nn_impl<   CosineDistance>(data, k);
	}
	__builtin_unreachable();
}


sp_mat
dmcore::nn_to_mat(Mat<size_t> neighbors, mat distances, size_t k) {
	assert(k <= neighbors.n_rows);
	assert(neighbors.n_rows == distances.n_rows);
	assert(neighbors.n_cols == distances.n_cols);
	sp_mat sp_dists(neighbors.n_cols, neighbors.n_cols);
	
	for (size_t i = 0; i < neighbors.n_cols; i++) {
		size_t* ns = neighbors.colptr(i);
		double* ds = distances.colptr(i);
		for (size_t j = 0; j < k; j++) {
			sp_dists.at(i, ns[j]) = ds[j];
			sp_dists.at(ns[j], i) = ds[j];
		}
	}
	
	return sp_dists;
}

sp_mat
dmcore::nn_to_mat(Mat<size_t> neighbors, mat distances) {
	return dmcore::nn_to_mat(neighbors, distances, neighbors.n_rows);
}

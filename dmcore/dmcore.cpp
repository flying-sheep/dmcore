#include "dmcore.hpp"

#include <cstdint>
#include <tuple>

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
std::tuple<Mat<size_t>, mat>
get_nn_impl(mat data, size_t k) {
	NeighborSearch<NearestNeighborSort, MetricType, mat, StandardCoverTree> nn(data);
	
	Mat<size_t> neighbors;
	mat distances;
	nn.Search(k, neighbors, distances);
	
	return std::make_tuple(neighbors, distances);
}

std::tuple<Mat<size_t>, mat>
dmcore::get_nn(mat data, size_t k, DistanceMetric metric) {
	switch(metric) {
		case DistanceMetric::Euclidean: return get_nn_impl<EuclideanDistance>(data, k);
		case DistanceMetric::Cosine:    return get_nn_impl<   CosineDistance>(data, k);
	}
	__builtin_unreachable();
}

sp_mat
dmcore::nn_to_mat(Mat<size_t> neighbors, mat distances) {
	sp_mat sp_dists(neighbors.n_cols, neighbors.n_cols);
	
	for (size_t i = 0; i < neighbors.n_cols; i++) {
		Mat<size_t>::const_col_iterator n = neighbors.begin_col(i);
		Mat<double>::const_col_iterator d = distances.begin_col(i);
		const Mat<size_t>::const_col_iterator n_end = neighbors.end_col(i);
		for (; n != n_end; n++, d++) {
			sp_dists(i, *n) = *d;
			sp_dists(*n, i) = *d;
		}
	}
	
	return sp_dists;
}

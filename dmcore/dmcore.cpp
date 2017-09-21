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

#include "dmcore.hpp"

#include <cstdint>

#include <mlpack/core.hpp>
#include <mlpack/core/tree/cover_tree/typedef.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <armadillo>

using std::size_t;

using mlpack::data::Load;
using mlpack::neighbor::NeighborSearch;
using mlpack::neighbor::NearestNeighborSort;
using mlpack::metric::EuclideanDistance;
using mlpack::tree::StandardCoverTree;

size_t test() {
	arma::mat data;
	Load("data.csv", data, true);
	NeighborSearch<NearestNeighborSort, EuclideanDistance, arma::mat, StandardCoverTree> nn(data);
	
	arma::Mat<size_t> neighbors;
	arma::mat distances;
	nn.Search(1, neighbors, distances);
	
	for (size_t i = 0; i < neighbors.n_elem; ++i) {
		std::cout << "Nearest neighbor of point " << i << " is point " << neighbors[i] << " and the distance is " << distances[i] << ".\n";
	}
	return neighbors.n_elem;
}

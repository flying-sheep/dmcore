#include "dmcore.hpp"

#include <iostream>

#include <mlpack/core.hpp>

using arma::mat;
using arma::Mat;
using arma::sp_mat;

using mlpack::data::Load;

int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::cerr << "Missing argument" << std::endl;
		return 1;
	}
	
	mat data;
	Load(argv[1], data, true);
	
	Mat<size_t> neighbors;
	mat distances;
	std::tie(neighbors, distances) = dmcore::get_nn(data, 5, dmcore::DistanceMetric::Euclidean);
	
	sp_mat sp_dist = dmcore::nn_to_mat(neighbors, distances);
	
	std::cout << sp_dist << std::endl;
	return 0;
}

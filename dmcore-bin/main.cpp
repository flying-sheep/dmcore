#include "dmcore.hpp"

#include <iostream>

#include <mlpack/core.hpp>

using arma::mat;
using arma::Mat;

using mlpack::data::Load;

int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::cout << "Missing argument" << std::endl;
		return 1;
	}
	
	mat data;
	Load(argv[1], data, true);
	
	Mat<size_t> neighbors;
	mat distances;
	std::tie(neighbors, distances) = get_nn(data, 5, dmcore::DistanceMetric::Euclidean);
	
	std::cout << neighbors << std::endl << distances << std::endl;
	return 0;
}

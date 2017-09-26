#include "dmcore.hpp"

#include <iostream>

#include <mlpack/core.hpp>

using arma::mat;
using arma::Mat;
using arma::sp_mat;
using arma::SpSubview;

using mlpack::data::Load;

sp_mat sp_diagmat(vec diag) {
	sp_mat dmat(diag.n_elem, diag.n_elem);
	dmat.diag() = diag;
	return dmat;
}

mat peek(sp_mat m) {
	return mat(sp_mat(m.head_cols(10)).head_rows(10));
}

int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::cerr << "Missing argument" << std::endl;
		return 1;
	}
	
	mat data;
	Load(argv[1], data, true);
	
	const auto nd = dmcore::get_nn(data, 30, dmcore::DistanceMetric::Euclidean);
	const Mat<size_t> neighbors = std::get<0>(nd);
	const mat         distances = std::get<1>(nd);
	
	const sp_mat dists = dmcore::nn_to_mat(neighbors, distances);
	std::cout << "dists:\n" << peek(dists) << std::endl;
	
	const vec sigmas = dmcore::local_sigmas(dists, 4, 6);
	std::cout << "sigmas:\n" << sigmas.head(10) << std::endl;
	const sp_mat trans_p = dmcore::transition_probabilities(dists, sigmas);
	std::cout << "trans_p:\n" << peek(trans_p) << std::endl;
	
	const vec d = vec(sum(trans_p, 1)) + 1; // diagonal set to 1
	
	// normalise by density
	const sp_mat norm_p = dmcore::get_norm_p(trans_p, d);
	std::cout << "norm_p:\n" << peek(norm_p) << std::endl;
	
	const vec d_norm = vec(sum(norm_p, 1));
	
	// calculate the inverse of a diagonal matrix by inverting the diagonal
	const sp_mat d_rot = sp_diagmat(pow(d_norm, -.5));
	const sp_mat transitions = d_rot * norm_p * d_rot;
	
	std::cout << "transitions:\n" << peek(transitions) << std::endl;
	return 0;
}

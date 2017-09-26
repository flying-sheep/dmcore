#include "dmcore.hpp"

#include <cstdint>
#include <utility>
#include <fstream>

#include <mlpack/core.hpp>
#include <mlpack/core/tree/cover_tree/typedef.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <armadillo>

using namespace dmcore;

using std::size_t;

using arma::vec;
using arma::mat;
using arma::Mat;
using arma::sp_mat;
using arma::SpSubview;
using arma::vectorise;

using mlpack::neighbor::NeighborSearch;
using mlpack::neighbor::NearestNeighborSort;
using mlpack::metric::EuclideanDistance;
using mlpack::kernel::CosineDistance;
using mlpack::tree::StandardCoverTree;


template<typename MetricType>
std::pair<Mat<size_t>, mat>
get_nn_impl(const mat data, const size_t k) {
	NeighborSearch<NearestNeighborSort, MetricType, mat, StandardCoverTree> nn(data);
	
	std::pair<Mat<size_t>, mat> nd;
	nn.Search(k, nd.first, nd.second);
	
	return nd;
}

std::pair<Mat<size_t>, mat>
dmcore::get_nn(const mat data, const size_t k, const DistanceMetric metric) {
	switch(metric) {
		case DistanceMetric::Euclidean: return get_nn_impl<EuclideanDistance>(data, k);
		case DistanceMetric::Cosine:    return get_nn_impl<   CosineDistance>(data, k);
	}
	__builtin_unreachable();
}


sp_mat
dmcore::nn_to_mat(const Mat<size_t> neighbors, const mat distances, size_t k) {
	assert(k <= neighbors.n_rows);
	assert(neighbors.n_rows == distances.n_rows);
	assert(neighbors.n_cols == distances.n_cols);
	sp_mat sp_dists(neighbors.n_cols, neighbors.n_cols);
	
	for (size_t i = 0; i < neighbors.n_cols; i++) {
		const size_t* ns = neighbors.colptr(i);
		const double* ds = distances.colptr(i);
		for (size_t j = 0; j < k; j++) {
			sp_dists.at(i, ns[j]) = ds[j];
			sp_dists.at(ns[j], i) = ds[j];
		}
	}
	
	std::ofstream myfile;
	myfile.open("coords.txt");
	const sp_mat::const_iterator start = sp_dists.begin();
	const sp_mat::const_iterator end   = sp_dists.end();
	for (sp_mat::const_iterator it = start; it != end; ++it) {
		myfile << it.col() << "\t" << it.row() << "\t" << *it << std::endl;
	}
	myfile.close();
	
	return sp_dists;
}

sp_mat
dmcore::nn_to_mat(const Mat<size_t> neighbors, const mat distances) {
	return dmcore::nn_to_mat(neighbors, distances, neighbors.n_rows);
}


sp_mat
dmcore::transition_probabilities(const sp_mat dists, const double sigma) {
	assert(dists.n_cols == dists.n_rows);
	const double s1 = pow(sigma, 2);
	const double s2 = 2*sigma;
	
	sp_mat t_p(dists);
	
	const sp_mat::iterator start = t_p.begin();
	const sp_mat::iterator end   = t_p.end();
	for (sp_mat::iterator it = start; it != end; ++it) {
		*it = sqrt(2 * s1 / s2) * exp(-pow(*it, 2) / s2);
	}
	
	t_p.diag().zeros();
	return t_p;
}

sp_mat
dmcore::transition_probabilities(const sp_mat dists, const vec sigmas) {
	assert(sigmas.n_elem == dists.n_rows);
	assert(dists.n_cols == dists.n_rows);
	const vec sigsp2 = pow(sigmas, 2);
	
	sp_mat t_p(dists);
	
	const sp_mat::iterator start = t_p.begin();
	const sp_mat::iterator end   = t_p.end();
	for (sp_mat::iterator it = start; it != end; ++it) {
		const double s1 = sigmas[it.col()] * sigmas[it.row()];
		const double s2 = sigsp2[it.col()] + sigsp2[it.row()];
		*it = sqrt(2 * s1 / s2) * exp(-pow(*it, 2) / s2);
		// std::cout << it.row() << " " << it.col() << std::endl;
	}
	
	t_p.diag().zeros();
	
	return t_p;
}

vec
dmcore::local_sigmas(const sp_mat dists, const size_t n_local_start, const size_t n_local_end) {
	const SpSubview<double> sig_mat = dists.cols(n_local_start, n_local_end);
	const vec row_sums(sum(sig_mat, 1));
	return row_sums / sig_mat.n_cols / 2;
}

vec
dmcore::local_sigmas(const sp_mat dists, const size_t n_local) {
	return dmcore::local_sigmas(dists, n_local, n_local);
}


sp_mat
dmcore::get_norm_p(const sp_mat trans_p, const vec d, const vec d_new) {
	assert(d.n_elem == d_new.n_elem);
	assert(d.n_elem == trans_p.n_cols);
	assert(d.n_elem == trans_p.n_rows);
	
	sp_mat norm_p(trans_p);
	const sp_mat::iterator start = norm_p.begin();
	const sp_mat::iterator end   = norm_p.end();
	for (sp_mat::iterator it = start; it != end; ++it) {
		*it = *it / (d_new[it.row()] * d[it.col()]);
	}
	return norm_p;
}

sp_mat
dmcore::get_norm_p(const sp_mat trans_p, const vec d) {
	return dmcore::get_norm_p(trans_p, d, d);
}

imputed_data <- read.delim('guo_raw.tsv')

sigma <- 'local'
distance <- 'euclidean'
n <- nrow(imputed_data)
k <- 5
n_local <- 3:5
dists <- NULL
verbose <- FALSE

knn <- destiny:::find_knn(imputed_data, dists, k, verbose)

sigmas <- get_sigmas(imputed_data, knn$nn_dist, sigma, n_local, distance, censor_val, censor_range, missing_range, vars, verbose)
sigma <- optimal_sigma(sigmas)  # single number = global, multiple = local

trans_p <- transition_probabilities(imputed_data, sigma, distance, knn$dist, censor, censor_val, censor_range, missing_range, verbose)
rm(knn)  # free memory

d <- rowSums(trans_p, na.rm = TRUE) + 1 # diagonal set to 1

# normalize by density if requested
norm_p <- get_norm_p(trans_p, d, d, density_norm)
rm(trans_p)  # free memory

d_norm <- rowSums(norm_p)

# calculate the inverse of a diagonal matrix by inverting the diagonal
d_rot <- Diagonal(x = d_norm ^ -.5)
transitions <- as(d_rot %*% norm_p %*% d_rot, 'symmetricMatrix')
rm(norm_p)  # free memory

eig_transitions <- decomp_transitions(transitions, n_eigs, verbose)

eig_vec <- eig_transitions$vectors
if (rotate) eig_vec <- as.matrix(t(t(eig_vec) %*% d_rot))
colnames(eig_vec) <- paste0('DC', seq(0, n_eigs))

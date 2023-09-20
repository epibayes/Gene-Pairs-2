data {
    int N; // number of individuals
    int M; // number of pairs. This should be n choose 2
    int P; // number of paired features (e.g. difference)
    int Q; // number of individual-level features
    vector[M] y; // log patristic distance
    matrix[M, P] X; // paired features
    matrix[M, Q] Z; // individual features
    matrix [M, N] W; // pairs matrix (sparse)
    matrix[N, N] S; // spatial distances

}

transformed data {
    vector[N] zeros = rep_vector(0, N);
}

parameters {
    vector[P] beta; // coefficients on pairwise differences
    vector[Q] gamma; // coefficients on individual features
    # vector[N] theta; // spatially-referenced random effects
    real<lower=0> sigma2_eps; // variance of noise
    real<lower=0> tau2; // baseline GP kernel variance
    real<lower=0> sigma2_zeta; // individual effect variance
    real<lower=0> ell; // GP kernel length scale
    vector[N] zeta;
    vector[N] eta;

}


model {
    # consider changing prior
    sigma2_eps ~ inv_gamma(0.01, 0.01);
    tau2 ~ inv_gamma(0.01, 0.01);
    sigma2_zeta ~ inv_gamma(0.01, 0.01);
    ell ~ inv_gamma(1, 1);

    matrix[N, N] K;
    for (i in 1:(N-1)) {
        K[i, i] = tau2;
        for (j in (i+1):N) {
            K[i, j] = exp(- S[i,j] / ell);
            K[j, i] = K[i, j];
        }
    }
    K[N, N] = tau2;

    matrix[N, N] L_K;
    L_K = cholesky_decompose(K);

    zeta ~ normal(0, sigma2_zeta);
    eta ~ multi_normal_cholesky(zeros, L_K);


    for (p in 1:P)
        beta[p] ~ normal(0, 100^2);
    for (q in 1:Q)
        gamma[q] ~ normal(0, 100^2);

    vector[N] theta;
    theta = eta + zeta;


    vector[M] mu;
    # todo: replace with pure matrix algebra
    mu = X * beta + Z * gamma + W * theta;


    y ~ normal(mu, sigma2_eps);
    
}


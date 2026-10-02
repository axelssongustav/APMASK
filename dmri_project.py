"""
===============================================================================
Bayesian Inference in Diffusion Tensor Imaging - Project Template
===============================================================================

This Python file provides the starter template for the course project in
"Advanced Probabilistic Machine Learning",
Department of Information Technology, Uppsala University.

Authors:
- Jens Sjölund (original author) - jens.sjolund@it.uu.se
- Anton O'Nils (updates & finalization) - anton.o-nils@it.uu.se
- Stina Brunzell (updates & finalization) - stina.brunzell@it.uu.se

------------------------------------------------------------------------------
Purpose
------------------------------------------------------------------------------
The project concerns Bayesian inference in diffusion MRI (dMRI), specifically
the diffusion tensor model (DTI). The goal is to estimate local tissue
properties (baseline signal S0 and diffusion tensor D) from real-world dMRI
measurements, using different Bayesian inference techniques.

Each student/group member will implement one of the following inference methods:
  1. Metropolis-Hastings  
  2. Importance Sampling  
  3. Variational Inference  
  4. Laplace Approximation  

The provided code gives:
  - Utilities for loading and preprocessing the "Stanford HARDI dataset". 
  - Helper functions for matrix operations, parameterizations and gradients.  
  - A skeleton structure for the prior, likelihood, and posterior approx.   
  - Placeholders where each inference method should be implemented.  
  - Plotting routines to visualize posterior summaries.

------------------------------------------------------------------------------
Dataset
------------------------------------------------------------------------------
The code uses the Stanford HARDI diffusion MRI dataset (Rokem et al., 2015),
accessible via DIPY's "get_fnames('stanford_hardi')".

------------------------------------------------------------------------------
Notes
------------------------------------------------------------------------------
- Several classes and methods are left as "NotImplementedError"; students are
  expected to fill these in.  
- Computations are memoized with "disk_memoize" to avoid repeated costly runs.  
- Results for each inference method are automatically plotted and saved.  

=============================================================================
Imports
=============================================================================
Required libraries: numpy, matplotlib, scipy, dipy
Install with: pip install numpy matplotlib scipy dipy
"""

# Standard library: general utilities
import os
import pickle
import hashlib
from functools import wraps


# NumPy and Matplotlib: math and plotting
import numpy as np
import matplotlib.pyplot as plt

# SciPy: probability distributions, math functions, and optimization
# Hint: these tools might be useful later in the project
from scipy.stats import gamma, norm, wishart, multivariate_normal
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp, digamma
from scipy.optimize import minimize

# DIPY: diffusion MRI utilities and models
from dipy.io.image import load_nifti, save_nifti   # for loading / saving imaging datasets
from dipy.io.gradients import read_bvals_bvecs     # for loading / saving our bvals and bvecs
from dipy.core.gradients import gradient_table     # for constructing gradient table from bvals/bvecs
from dipy.data import get_fnames                   # for small datasets that we use in tests and examples
from dipy.segment.mask import median_otsu          # for masking out the background
import dipy.reconst.dti as dti                     # for diffusion tensor model fitting and metrics



"""
=============================================================================
Caching Utility (already implemented)
=============================================================================
Provides disk-based memoization to avoid recomputation.
"""

def disk_memoize(cache_dir="cache"):
    """
    Decorator for caching function outputs on disk.

    This utility is already implemented and should not be modified by students.
    It allows expensive computations to be stored and re-used across runs,
    based on the function arguments. If you call the same function again with
    the same inputs, it returns the cached results instead of recomputing.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Optionally force a fresh computation (ignores cache if True)
            force = kwargs.pop("force_recompute", False)

            # Make sure the cache directory exists
            os.makedirs(cache_dir, exist_ok=True)

            # Build a unique hash key from the function name and arguments
            func_name = func.__name__
            key = (func_name, args, kwargs)
            hash_str = hashlib.md5(pickle.dumps(key)).hexdigest()
            cache_path = os.path.join(cache_dir, f"{func_name}_{hash_str}.pkl")

            # Load the cached result if it exists (and recomputation is not forced)
            if not force and os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

            # Otherwise: compute the result, then cache it to disk
            result = func(*args, **kwargs)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)

            return result
        
        return wrapper
    return decorator



"""
=============================================================================
Data Loading & Preprocessing (already implemented)
=============================================================================
Loads the Stanford HARDI dataset, applies masking/cropping, and
extracts one voxel with a DTI point estimate for testing.
"""

@disk_memoize()
def get_preprocessed_data():
    """
    Load and preprocess a single voxel of diffusion MRI data.

    What it does:
    - Loads the dataset and gradient information (b-values and b-vectors).
    - Fits a diffusion tensor model (DTI) to one voxel.
    - Extracts a point estimate: baseline signal (S0), eigenvalues, eigenvectors.

    Returns
    -------
    y : ndarray
        Observed diffusion MRI signal vector for a single voxel.
    point_estimate : [S0, evals, evecs]
        Estimated baseline signal, eigenvalues, and eigenvectors.
    gtab : GradientTable
        Gradient table with b-values (diffusion weighting strength)
        and b-vectors (gradient directions).
    """

    # Load the masked data, background mask, and gradient information
    data, mask, gtab = get_data()

    # Initialize a diffusion tensor model (DTI) with S0 estimation enabled
    tenmodel = dti.TensorModel(gtab, return_S0_hat=True)

    # Extract the signal for a single voxel (coordinates chosen for this project)
    y = data[35, 35, 30, :]

    # Fit the DTI model to this voxel's signal
    tenfit = tenmodel.fit(y)
    
    # Extract point estimates: baseline signal, eigenvalues, and eigenvectors
    S0 = tenfit.S0_hat
    evals = tenfit.evals
    evecs = tenfit.evecs
    point_estimate = [S0, evals, evecs]

    # Return the raw voxel signal, point estimate, and gradient table
    return y, point_estimate, gtab


def get_data():
    """
    Load and preprocess the Stanford HARDI diffusion MRI dataset.

    What it does:
    - Downloads the dataset if not already present (via DIPY).
    - Loads the 4D diffusion MRI volume (x, y, z, measurements).
    - Reads b-values (diffusion weighting strength) and b-vectors (gradient directions).
    - Creates a gradient table (gtab) combining this information.
    - Applies a brain mask and cropping to remove background and reduce size.

    Returns
    -------
    maskdata : ndarray
        The masked and cropped diffusion MRI data.
    mask : ndarray (boolean)
        The brain mask used to exclude background voxels.
    gtab : GradientTable
        Gradient information (b-values and b-vectors) for each measurement.
    """

    # Download filenames for the Stanford HARDI dataset if not already cached 
    hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

    # Load the raw 4D dataset: dimensions are (x, y, z, diffusion measurements)
    data, _ = load_nifti(hardi_fname)

    # Read diffusion weighting information (b-values, b-vectors) and build gradient table
    bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
    gtab = gradient_table(bvals, bvecs)

    # Apply brain masking and cropping to remove background and save compute
    maskdata, mask = median_otsu(
        data, vol_idx=range(10, 50), median_radius=3, numpass=1, autocrop=True, dilate=2
    )

    # Print the final data shape for confirmation
    print('Loaded data with shape: (%d, %d, %d, %d)' % maskdata.shape)

    return maskdata, mask, gtab


"""
=============================================================================
Linear Algebra Helpers (already implemented)
=============================================================================
Functions for reconstructing tensors and switching between
parameterizations. Already implemented.
Hint: you will make use of these helpers later in the project,
the ones involving theta are useful for VI and Laplace.
"""

def compute_D(evals, V):
    """
    Reconstruct the diffusion tensor D from eigenvalues and eigenvectors.

    D = V Λ V.T, where Λ is the diagonal matrix of eigenvalues.

    Parameters
    ----------
    evals : ndarray
        Eigenvalues, shape (3,) or batched.
    V : ndarray
        Eigenvectors, shape (3, 3) or batched.

    Returns
    -------
    D : ndarray
        Diffusion tensor(s), shape (..., 3, 3).
    """

    # Ensure inputs have the correct batch dimensions
    if evals.ndim == 1:
        evals = evals[None, None, :]
    elif evals.ndim == 2:
        evals = evals[:, None, :]
    if V.ndim == 2:
        V = V[None, :, :]

    # Compute D = V Λ V.T as V (V @ Λ).T
    V_scaled = V * evals
    D = np.matmul(V, np.transpose(V_scaled, axes=[0, 2, 1]))

    return D


def theta_from_D(D):
    """
    Convert a diffusion tensor D into an unconstrained parameter vector theta.

    Follows Eq. (18): D = L L.T with L from Cholesky factorization.
    Diagonals are log-transformed, off-diagonals kept raw.

    Parameters
    ----------
    D : ndarray (3, 3)
        Symmetric positive-definite diffusion tensor.

    Returns
    -------
    theta : ndarray (6,)
        Unconstrained parameter vector corresponding to the lower-triangular
        entries of L (log of diagonals, raw off-diagonals).
    """
    
    # Compute Cholesky factor (lower-triangular L) of D
    L = np.linalg.cholesky(D)
    
    # Indices of lower-triangular entries (including diagonal)
    p = D.shape[0]
    tril_indices = np.tril_indices(p)
    theta = []

    # Store log of diagonal entries, raw off-diagonal entries
    for i, j in zip(*tril_indices):
        if i == j:
            theta.append(np.log(L[i, j]))   # Diagonal: log-transform
        else:
            theta.append(L[i, j])           # Off-diagonal: raw value

    return np.array(theta)


def D_from_theta(theta):
    """
    Convert unconstrained parameter vector theta back into diffusion tensor D.

    Follows Eq. (18): D = L L.T with L constructed from theta.
    Diagonal entries of L are exponentiated to ensure positivity,
    off-diagonals are used as raw values.

    Parameters
    ----------
    theta : ndarray (..., 6)
        Unconstrained parameters corresponding to the lower-triangular
        entries of L (log-diagonals, raw off-diagonals).

    Returns
    -------
    D : ndarray (..., 3, 3)
        Symmetric positive-definite diffusion tensor(s).
    """
    
    # Ensure theta is an array and check shape
    theta = np.asarray(theta)
    *batch_shape, _ = theta.shape
    assert theta.shape[-1] == 6, "Last dimension must be 6 for 3x3 lower-triangular matrices."

    # Initialize lower-triangular matrix L
    L = np.zeros((*batch_shape, 3, 3), dtype=theta.dtype)

    # Fill L with exponentiated diagonals and raw off-diagonals
    tril_indices = np.tril_indices(3)
    for k, (i, j) in enumerate(zip(*tril_indices)):
        if i == j:
            L[..., i, j] = np.exp(theta[..., k])   # Diagonal
        else:
            L[..., i, j] = theta[..., k]           # Off-diagonal

    # Reconstruct D = L @ L.T (batch-aware matrix multiplication)
    D = L @ np.swapaxes(L, -1, -2)

    return D.squeeze()


def grad_D_wrt_theta_at_D(D):
    """
    Compute nabla_theta D evaluated at D.

    Uses the parameterization in Eq. (18): D = L L.T where L is built from 
    theta. Returns the gradient tensor with one (3x3) slice per theta component.

    Parameters
    ----------
    D : ndarray (3, 3)
        Symmetric positive-definite diffusion tensor.

    Returns
    -------
    grad_D : ndarray (3, 3, 6)
        Gradient of D w.r.t. theta, one 3x3 matrix per parameter.
    """
    
    # Get Cholesky factor of D and set up indices for lower-triangular entries
    p = D.shape[0]
    L = np.linalg.cholesky(D)
    tril_indices = np.tril_indices(p)
    num_params = len(tril_indices[0])

    # Prepare output container
    grad_D = np.zeros((p, p, num_params))

    # Loop over all parameters in theta
    for k, (m, n) in enumerate(zip(*tril_indices)):

        # Build a basis matrix for the effect of this parameter
        E_mn = np.zeros((p, p))
        if m == n:
            # Diagonal: dL_mm/dtheta = L_mm since L_mm = exp(theta)
            factor = L[m, n]
        else:
            # Off-diagonal: dL_mn/dtheta = 1
            factor = 1.0
        E_mn[m, n] = factor

        # Work out the corresponding change in D
        dD_k = E_mn @ L.T + L @ E_mn.T
        grad_D[:, :, k] = dD_k

    return grad_D


"""
=============================================================================
Bayesian Model Components (need to be implemented)
=============================================================================
Students: implement all parts in this section (priors, likelihoods, etc.)
These are required before any inference method can be attempted.
"""

class frozen_prior:

    def __init__(self, alpha_S = 2, theta_S = 500, alpha_lam = 4, theta_lam = 2.5e-4):
        self.S0_prior = gamma(a=alpha_S, scale=theta_S)
        self.lam_prior = gamma(a=alpha_lam, scale = theta_lam)
    
    def rvs(self, size = 1):
        S0 = self.S0_prior.rvs(size=size) #Shape (size,)
        evals = self.lam_prior.rvs(size=(size,3)) #Shape (size, 3)
        evecs = Rotation.random(size).as_matrix() #Shape (size, 3, 3)
        return S0, evals, evecs
    
    def logpdf(self, S0, evals):
        return self.S0_prior.logpdf(S0) + np.sum(self.lam_prior.logpdf(evals), axis = -1)

class frozen_likelihood:


    def __init__(self, gtab, y, sigma=29):
        self.gtab = gtab   # store gradient table with b-values and b-vectors
        self.y = np.asarray(y, dtype=float)   # measured signal of our voxel, shape (N,)
        self.sigma = sigma                    # noise standard deviation, Table 1

    def logpdf(self, S0, evecs, evals):
        S0 = np.atleast_1d(S0)        # ensure S0 is array-like
        D = compute_D(evals, evecs)   # reconstruct diffusion tensor

        # Build q from diffusion gradients (b-values & b-vectors),
        # corresponds to the experimental setting x in the project instructions
        q = np.sqrt(self.gtab.bvals[:, None]) * self.gtab.bvecs

        # Model signal S given tensor D and baseline S0
        S = S0[:, None] * np.exp( - np.einsum('...j, ijk, ...k->i...', q, D, q))

        #Gaussian log density of each measurement, summed over measurements, shape (B,)
        return np.sum(norm.logpdf(self.y, loc=S, scale=self.sigma), axis=1)

    def rvs(self, S0, evecs, evals, size=1):
        S0 = np.atleast_1d(S0)
        D = compute_D(evals, evecs)
        q = np.sqrt(self.gtab.bvals[:, None]) * self.gtab.bvecs
        S = S0[:, None] * np.exp( - np.einsum('...j, ijk, ...k->i...', q, D, q))

        #Predicted signal plus Gaussian noise, shape (size, B, N)
        return norm.rvs(loc=S, scale=self.sigma, size=(size,) + S.shape)

def check_model():
    #Check the prior and likelihood before any inference method uses them
    y, point_estimate, gtab = get_preprocessed_data()
    S0_hat, evals_hat, evecs_hat = point_estimate
    prior = frozen_prior()
    likelihood = frozen_likelihood(gtab, y)

    #For prior, sample means should match the theoretical means (Gamma mean = shape * scale)
    S0, evals, evecs = prior.rvs(size=100000)
    print("Prior mean of S0:", S0.mean(), " theory:", 2 * 500)
    print("Prior mean of eigenvalues:", evals.mean(axis=0), " theory:", 4 * 2.5e-4)
    print("Prior mean of D (should be close to 0.001 * identity):")
    print(compute_D(evals, evecs).mean(axis=0))

    #Predicted signal at DIPY's estimate, Eq. (1)
    D_hat = compute_D(evals_hat, evecs_hat)[0]
    q = np.sqrt(gtab.bvals[:, None]) * gtab.bvecs
    S_hat = S0_hat * np.exp(-np.sum((q @ D_hat) * q, axis=1))

    #For likelihood, simulated data should average to the predicted signal, with spread sigma
    y_sim = likelihood.rvs(S0_hat, evecs_hat, evals_hat, size=10000)[:, 0, :]
    print("Largest gap between mean of simulated y and predicted signal:",
          np.max(np.abs(y_sim.mean(axis=0) - S_hat)), " (should be about 1 or less)")
    print("Std of simulated y:", y_sim.std(axis=0).mean(), " theory:", 29)

    #(Likelihood) The real data should look like the prediction plus noise of size sigma
    print("Residual std at DIPY fit:", (y - S_hat).std(), " sigma in model:", 29)

    #(Likelihood) DIPY's fit
    ll_hat = likelihood.logpdf(S0_hat, evecs_hat, evals_hat)[0]
    ll_random = likelihood.logpdf(S0[:1000], evecs[:1000], evals[:1000])
    print("Log-likelihood at DIPY fit:", ll_hat, " best of 1000 prior draws:", ll_random.max())
    print("Log prior at DIPY fit:", prior.logpdf(S0_hat, evals_hat))

"""
=============================================================================
Posterior Approximations (need to be implemented)
=============================================================================
Students: implement these approximations, which are only used in the
corresponding inference methods below:
  - variational_posterior: used only for Variational Inference
  - mvn_reparameterized: used only for Laplace Approximation

They are NOT needed for Metropolis-Hastings or Importance Sampling.
"""

class variational_posterior:
    # Placeholder for variational posterior approximation.
    # Hint: you may want to add input parameters to these methods.
    # The score() method is already implemented and can be used later
    # when implementing inference (with REINFORCE leave-one-out estimator).

    def __init__(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (9,):
            raise ValueError("Theta needs shape 9")
        self.theta = theta.copy()

        #q(SO), gamma(alpha/shape, beta/scale)
        self.shape = np.exp(theta[0])
        self.scale = np.exp(theta[1])

        #q(D), Wishart params, call mean matrix Sigma
        self.Sigma = D_from_theta(theta[2:8])
        self.df = np.exp(theta[8]) + 2.0

    def logpdf(self, S0, D):
        #dimension check/fix so all is in correct shape
        S0 = np.atleast_1d(S0)
        D = np.asarray(D)
        if D.ndim == 2:
            D = D[None, :, :]
        
        #define the logs
        log_q_S0 = gamma.logpdf(S0, a = self.shape, scale = self.scale)

        q_D = wishart(df = self.df, scale = self.Sigma/self.df)
        log_q_D = np.array([q_D.logpdf(D_i) for D_i in D])

        return log_q_S0 + log_q_D

    def rvs(self, size):
        #sample S0
        S0_samples = gamma.rvs(a = self.shape, scale = self.scale, size = size)

        #sample diffusion tensors
        D_samples = wishart.rvs(self.df, scale = self.Sigma/self.df, size = size)

        #make sure we get the dimensions right
        D_samples = np.asarray(D_samples)
        if D_samples.ndim == 2:
            D_samples = D_samples[None, :, :]

        #compute eigenvalues of samples, use fact D symmetric
        evals_samples, evecs_samples = np.linalg.eigh(D_samples)

        return S0_samples, evals_samples, evecs_samples

    def score(self, S0, D):
        # Combine score contributions from gamma and Wishart parts
        score_wrt_log_shape, score_wrt_log_scale = self.gamma_score(S0)
        score_wrt_theta, score_wrt_log_df = self.wishart_score(D)
        return np.concatenate([
            np.atleast_1d(score_wrt_log_shape), np.atleast_1d(score_wrt_log_scale), np.atleast_1d(score_wrt_theta), np.atleast_1d(score_wrt_log_df)]
        )

    def gamma_score(self, x):
        # Score function for gamma distribution
        score_wrt_log_shape = (np.log(x / self.scale) - digamma(self.shape)) * self.shape
        score_wrt_log_scale = (x / self.scale**2 - self.shape / self.scale) * self.scale
        return score_wrt_log_shape, score_wrt_log_scale

    def wishart_score(self, D):
        # Score function for Wishart distribution
        W = self.df * D
        Sigma_inv = np.linalg.inv(self.Sigma)
        score_wrt_Sigma = 0.5 * Sigma_inv @ (W - self.df * self.Sigma) @ Sigma_inv
        score_wrt_theta = np.tensordot(
            score_wrt_Sigma, grad_D_wrt_theta_at_D(self.Sigma), axes=([0,1], [0,1])
        )
        p = W.shape[0]
        _, logdet_W = np.linalg.slogdet(W)
        _, logdet_Sigma = np.linalg.slogdet(self.Sigma)
        digamma_sum = np.sum([digamma((self.df + 1 - j) / 2.0) for j in range(1, p+1)])
        score_wrt_log_df = ((self.df - 2) / 2) * (logdet_W - p * np.log(2) - logdet_Sigma - digamma_sum)
        return score_wrt_theta, score_wrt_log_df


class mvn_reparameterized:
    # Placeholder for multivariate normal approximation.
    # Hint: you may want to add input parameters to these methods.
    
    def __init__(self):
        raise NotImplementedError
    
    def rvs(self, size):
        raise NotImplementedError
    
        return S0_samples, evals_samples, evecs_samples


"""
=============================================================================
Inference Methods (need to be implemented)
=============================================================================
Students: implement one method each (MH, IS, VI, or Laplace).
Uses memoization to speed up repeated runs.
"""

@disk_memoize()
def metropolis_hastings(*args, **kwargs):
    # Students: implement Metropolis-Hastings here.
    # Before starting, make sure the prior and likelihood are implemented.
    # Note: you may change, add, or remove input parameters depending on your design
    # (e.g. pass initialization values like those prepared in main()).

    # Lightweight placeholder implementation: draw samples around DTI point estimate
    n_samples = kwargs.get('n_samples', 2000)
    y, point_estimate, gtab = get_preprocessed_data()
    S0_init, evals_init, evecs_init = point_estimate

    S0_samples = np.random.normal(loc=S0_init, scale=max(1e-2, 0.05 * S0_init), size=n_samples)
    evals_samples = np.maximum(1e-9, np.random.normal(loc=evals_init, scale=0.05 * evals_init, size=(n_samples, 3)))
    # replicate evecs (no rotation noise for simplicity)
    evecs_samples = np.repeat(evecs_init[None, :, :], n_samples, axis=0)

    return S0_samples, evals_samples, evecs_samples


@disk_memoize()
def importance_sampling(*args, **kwargs):
    # Students: implement Importance Sampling here.
    # Before starting, make sure the prior and likelihood are implemented.
    # Note: you may change, add, or remove input parameters depending on your design
    # (e.g. pass initialization values like those prepared in main()).

    n_samples = kwargs.get('n_samples', 2000)
    gamma_S = kwargs.get('gamma_S', 0.02)
    nu = kwargs.get('nu', 4000)

    y, point_estimate, gtab = get_preprocessed_data()
    S0_init, evals_init, evecs_init = point_estimate
    D_init = compute_D(evals_init, evecs_init)[0]

    prior = frozen_prior()
    likelihood = frozen_likelihood(gtab, y)

    proposal_S0 = gamma(a=gamma_S**-2, scale=gamma_S**2 * S0_init)
    proposal_D = wishart(df=nu, scale=D_init / nu)

    S0_samples = proposal_S0.rvs(size=n_samples)
    D_samples = proposal_D.rvs(size=n_samples)
    evals_samples, evecs_samples = np.linalg.eigh(D_samples)

    log_q = proposal_S0.logpdf(S0_samples) + proposal_D.logpdf(np.moveaxis(D_samples, 0, -1))

    l1, l2, l3 = evals_samples[:, 0], evals_samples[:, 1], evals_samples[:, 2]
    log_jacobian = np.log(l2 - l1) + np.log(l3 - l1) + np.log(l3 - l2)
    log_prior = prior.logpdf(S0_samples, evals_samples) - log_jacobian
    log_lik = likelihood.logpdf(S0_samples, evecs_samples, evals_samples)

    log_w = log_prior + log_lik - log_q
    importance_weights = np.exp(log_w - logsumexp(log_w))

    return importance_weights, S0_samples, evals_samples, evecs_samples


<<<<<<< HEAD
def tune_importance_sampling():
    """Small, readable grid search for gamma_S and nu."""
    best_gamma = None
    best_nu = None
    best_ess = -np.inf

    for gamma_S in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]:
        for nu in [2000, 5000, 10000, 20000, 40000, 80000]:
            weights, _, _, _ = importance_sampling(
                force_recompute=True,
                n_samples=2000,
                gamma_S=gamma_S,
                nu=nu,
            )
            ess = 1.0 / np.sum(weights**2)

            if ess > best_ess:
                best_ess = ess
                best_gamma = gamma_S
                best_nu = nu

            print(f"gamma_S={gamma_S:.2f} nu={nu:7d} ESS={ess:8.1f}")

    print("BEST:", best_gamma, best_nu, "ESS=", round(best_ess, 1))
    return (best_gamma, best_nu), best_ess


def variational_inference(*args, **kwargs):
=======
@disk_memoize()
def variational_inference(n_iterations = 10000, K = 50, learning_rate = 5e-3, theta_init = None, seed = 0, verbose = True):
>>>>>>> origin/main
    # Students: implement Variational Inference here.
    # Before starting, make sure the prior, likelihood and variational_posterior are implemented.
    # Note: you may change, add, or remove input parameters depending on your design
    # (e.g. pass initialization values like those prepared in main()).

    #initialize all we need
    np.random.seed(seed)
    y, point_estimate, gtab = get_preprocessed_data()
    S0_init, evals_init, evecs_init = point_estimate

    S0_init = float(np.asarray(S0_init).squeeze())
    D_init = compute_D(evals_init, evecs_init).squeeze()

    prior = frozen_prior()
    likelihood = frozen_likelihood(gtab, y)

    #get the nine variational parameters, draw them if none are given
    if theta_init is None:
        initial_gamma_shape = 10
        initial_gamma_scale = S0_init/initial_gamma_shape
        initial_df = 10

        theta = np.concatenate([
            np.array([
                np.log(initial_gamma_shape),
                np.log(initial_gamma_scale)]), #convert them back from exponents
                theta_from_D(D_init), #get thetas from D
                np.array([np.log(initial_df - 2.0)]) #again convert back
                ])
    else:
        theta = np.asarray(theta_init, dtpye = float).copy()
        #check that the given theta has right shape
        if theta.shape != (9,):
            raise ValueError("Theta needs correct shape (9,)")

    #setup Adam algorithm, parameter values taken from geeksforgeeks
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    first_moment = np.zeros_like(theta)
    second_moment = np.zeros_like(theta)

    #define so we can monitor the optimization progress
    elbo_history = []
    progress_every = max(1, n_iterations // 10)

    for iteration in range(1, n_iterations + 1):
        #distribution with current theta
        posterior = variational_posterior(theta)

        #draw K samples 
        S0_samples, evals_samples, evecs_samples = posterior.rvs(K)

        #compute D
        D_samples = compute_D(evals_samples, evecs_samples)

        #compute the log joint distribution
        log_joint = (
            likelihood.logpdf(S0_samples, evecs_samples, evals_samples) +
            prior.logpdf(S0_samples, evals_samples)
            )  

        #compute f_k = log p(y, z_k) - log q_theta(z_k)    
        log_q = posterior.logpdf(S0_samples, D_samples)    
        f_values = log_joint - log_q

        #keep a monte carlo estimate for monotoring
        elbo_history.append(np.mean(f_values))

        #compute scores, one vector per sample
        scores = np.stack([posterior.score(S0_samples[k], D_samples[k]) for k in range(K)])

        #leave one out baseline
        total_f = np.sum(f_values)
        baselines = (total_f - f_values) / (K-1)

        relative_f_values = f_values - baselines

        #reinforce leave one out estimator of gradient
        gradient = np.mean(relative_f_values[:, None] * scores, axis = 0)

        #perform Adam
        first_moment = (beta1 * first_moment) + ((1.0 - beta1) * gradient)

        second_moment = (beta2 * second_moment) + ((1 - beta2) * gradient**2)

        first_moment_corrected = first_moment / (1 - beta1**iteration)

        second_moment_corrected = second_moment / (1 - beta2**iteration)

        #use a + sign since we maximize
        theta += learning_rate * first_moment_corrected / (np.sqrt(second_moment_corrected) + epsilon)

        if (
            verbose
            and (
                iteration == 1
                or iteration % progress_every == 0
                or iteration == n_iterations
            )
        ):
            print(
                f"Iteration {iteration:5d}/{n_iterations}: "
                f"ELBO estimate = {elbo_history[-1]:.3f}"
            )

    posterior = variational_posterior(theta)
    posterior.elbo_history = np.asarray(elbo_history)

    return posterior

def laplace_approximation(*args, **kwargs):
    # Students: implement the Laplace Approximation here.
    # Before starting, make sure the prior, likelihood and mvn_reparameterized are implemented.
    # Note: you may change, add, or remove input parameters depending on your design
    # (e.g. pass initialization values like those prepared in main()).

    # Return a simple object with rvs(size) method producing samples similar to Laplace
    y, point_estimate, gtab = get_preprocessed_data()
    S0_init, evals_init, evecs_init = point_estimate

    class SimpleLaplace:
        def rvs(self, size):
            S0 = np.random.normal(loc=S0_init, scale=max(1e-2, 0.02 * S0_init), size=size)
            evals = np.maximum(1e-9, np.random.normal(loc=evals_init, scale=0.02 * evals_init, size=(size, 3)))
            evecs = np.repeat(evecs_init[None, :, :], size, axis=0)
            return S0, evals, evecs

    return SimpleLaplace()



"""
=============================================================================
Visualization & Experiment Runner
=============================================================================
Plotting function and the main() script to run experiments.
"""

def main():
    # Initialize with preprocessed data and DTI point estimate
    # (these values can be used as starting points for inference methods)
    y, point_estimate, gtab = get_preprocessed_data(force_recompute=False)
    S0_init, evals_init, evecs_init = point_estimate
    D_init = compute_D(evals_init, evecs_init).squeeze()

    # Find principal eigenvector from DTI estimate (for plotting)
    evec_principal = evecs_init[:, 0]

    # Set random seed and number of posterior samples
    np.random.seed(0)
    n_samples = 10000

    # # Run Metropolis–Hastings and plot results
    # S0_mh, evals_mh, evecs_mh = metropolis_hastings(force_recompute=False)
    # burn_in = 0
    # plot_results(S0_mh[burn_in:], evals_mh[burn_in:], evecs_mh[burn_in:, :, :], evec_principal, method="mh")

    # # Run Importance Sampling and plot results
    # w_is, S0_is, evals_is, evecs_is = importance_sampling(force_recompute=False)
    # plot_results(S0_is, evals_is, evecs_is, evec_principal, weights=w_is, method="is")

    # Run Variational Inference and plot results
    posterior_vi = variational_inference(force_recompute=True)
    S0_vi, evals_vi, evecs_vi = posterior_vi.rvs(size=n_samples)
    plot_results(S0_vi, evals_vi, evecs_vi, evec_principal, method="vi")


    # # Run Laplace Approximation and plot results
    # posterior_laplace = laplace_approximation(force_recompute=False)
    # S0_laplace, evals_laplace, evecs_laplace = posterior_laplace.rvs(size=n_samples)
    # plot_results(S0_laplace, evals_laplace, evecs_laplace, evec_principal, method="laplace")

    print("Done.")
    # Plot the ELBO after VI has finished
    elbo = posterior_vi.elbo_history

    plt.figure()
    plt.plot(elbo, alpha=0.3, label="Raw ELBO estimate")

    # Moving average to make the overall trend visible
    window = 100
    smoothed_elbo = np.convolve(
        elbo,
        np.ones(window) / window,
        mode="valid"
    )

    plt.plot(
        np.arange(window - 1, len(elbo)),
        smoothed_elbo,
        linewidth=2,
        label="100-iteration moving average"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Estimated ELBO")
    plt.title("Variational inference convergence")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("Convergence_VI.png", dpi=300, bbox_inches='tight')

def plot_results(S0, evals, evecs, evec_ref, weights=None, method=""):
    """
    Plot posterior results as histograms and save to file.

    Creates histograms of baseline signal (S0), mean diffusivity (MD),
    fractional anisotropy (FA), and the angle between estimated and
    reference eigenvectors.

    Parameters
    ----------
    S0 : ndarray
        Sampled baseline signals.
    evals : ndarray
        Sampled eigenvalues of the diffusion tensor.
    evecs : ndarray
        Sampled eigenvectors of the diffusion tensor.
    evec_ref : ndarray
        Reference principal eigenvector (from point estimate).
    weights : ndarray, optional
        Importance weights for samples. Uniform if None.
    method : str
        Name of inference method (used in output filename).
    """
    
    # Use uniform weights if none provided
    if weights is None:
        weights = np.ones_like(S0)
        weights /= np.sum(weights)

    # Choose number of bins based on sample size
    n_bins = np.floor(np.sqrt(len(weights))).astype(int)

    # Squeeze arrays for plotting
    weights = weights.squeeze()
    S0 = S0.squeeze()
    md = dti.mean_diffusivity(evals).squeeze()
    fa = dti.fractional_anisotropy(evals).squeeze()

    # Compute acute angle between estimated and reference eigenvectors
    angle = 360/(2*np.pi) * np.arccos(np.abs(np.dot(evecs[:, :, 2], evec_ref)))
    
    # Create 2x2 grid of histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharey=False)

    axes[0, 0].hist(S0, bins=n_bins, density=True, weights=weights, 
                    alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_xlabel("S0")
    axes[0, 0].set_ylabel("Density")

    axes[0, 1].hist(md, bins=n_bins, density=True, weights=weights, 
                    alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel("Mean diffusivity")
    axes[0, 1].set_ylabel("Density")

    axes[1, 0].hist(fa, bins=n_bins, density=True, weights=weights,
                     alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_xlabel("Fractional anisotropy")
    axes[1, 0].set_ylabel("Density")

    axes[1, 1].hist(angle, bins=n_bins, density=True, weights=weights, 
                    alpha=0.7, color='magenta', edgecolor='black')
    axes[1, 1].set_xlabel("Acute angle")
    axes[1, 1].set_ylabel("Density")

    # Adjust layout and save figure with method name
    plt.tight_layout()
    plt.savefig("results_{}.png".format(method), dpi=300, bbox_inches='tight')



if __name__ == "__main__":
    main()
    check_model()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 12:26:20 2023

@author: thosvarley


"""

#%% Imports 
import cython
cimport cython
from libc.math cimport log

import numpy as np # It's fine that both are imported as np.
cimport numpy as np

import networkx as nx 
from copy import deepcopy
from itertools import combinations, chain 
from scipy.stats import norm, multivariate_normal, linregress, pearsonr, zscore

# Setting up various numpy/C interface utils  
# Static typing to avoid overhead on type-changing operations. 
np.import_array()
DTYPE_double = np.float64
DTYPE_int = np.int64 

ctypedef np.float64_t DTYPE_double_t
ctypedef np.int64_t DTYPE_int_t

#%% Annoying boilerplate for the integrated information decomposition. 
lattice_orig = nx.read_gpickle("phi_lattice_22.pickle")    
distances = nx.shortest_path_length(lattice_orig, target = (((0,),(1,)),((0,),(1,))))

cdef list order = []
for distance in range(max(distances.values())+1):
    order += [key for key in distances.keys() if distances[key] == distance]

cdef set phir_atoms = { # Only used for the phi_r function.
    (((0,), (1,)), ((0, 1),)),
    (((1,),), ((0, 1),)),
    (((0, 1),), ((0,),)),
    (((0, 1),), ((0,), (1,))),
    (((0, 1),), ((1,),)),
    (((0, 1),), ((0, 1),)),
    (((0,),),((1,),)),
    (((1,),),((0,),)),
    }

#%% STATIC FUNCTIONS.

@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def local_entropy_1d(DTYPE_int_t idx1, 
                     np.ndarray[DTYPE_double_t, ndim=2] X):
    """
    Computes the local entropy for a one-dimensional, Gaussian variable. 
    
    h(x) = -log( P(x) ), where P(x) is assumed to be a univariate Gaussian. 

    Parameters
    ----------
    idx1 : int
        The index of the process to compute the local entropy for in X.
    X : np.ndarray 
        The multivariate time series array in cells x time format.

    Returns
    -------
    ents : np.ndarray
        The local entropy of X[idx1].

    """
    
    cdef DTYPE_double_t mu = X[idx1].mean() # Central tendency
    cdef DTYPE_double_t sigma = X[idx1].std() # Standard deviation
    # Magic call to scipy. 
    cdef np.ndarray[DTYPE_double_t, ndim=1] ents = -np.log(norm.pdf(X[idx1],
                                                                    loc=mu,
                                                                    scale=sigma)
                                                           )
    return ents

@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def local_entropy_nd(np.ndarray[DTYPE_double_t, ndim=2] X):
    """
    Computes the local entropy of the joint state of every row in X.
    
    Parameters
    ----------
    X : np.ndarray 
        The multivariate time series array in cells x time format.

    Returns
    -------
    np.ndarray
        The local entropy of X.

    """
    
    cdef np.ndarray[DTYPE_double_t, ndim=2] cov
    cdef np.ndarray[DTYPE_double_t, ndim=1] means
    cdef np.ndarray[DTYPE_double_t, ndim=1] ents
    
    # It gets grumpy if it's a 2D array with only one row
    # so in that case, we kick it to local_entropy_1d.
    if X.shape[0] == 1:
        return local_entropy_1d(0, X)
    else:
        cov = np.cov(X, ddof=0.0) # The covariance matrix. 
        means = X.mean(axis=-1) # The central tendencies 
        # Magic call to scipy. 
        ents = -np.log(multivariate_normal.pdf(X.T,
                                               mean=means, 
                                               cov=cov)
                       )
    return ents


@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.boundscheck(False)
def local_total_correlation(np.ndarray[DTYPE_double_t, ndim=2] X):
    """
    Returns a time series of the instantanious local total correlation 
    for every frame in X. Also called the integration by Tononi et al.,
    
    TC(X) is low when all X_i are independent. 
    TC(X) is high when all X_i in X are copies of each-other. 
    
    Total correlation is:
        
        tc(x) = sum( h(x_i) ) - h(x)
        
    See: 
        Tononi, G., Sporns, O., & Edelman, G. M. (1994). 
        A measure for brain complexity: Relating functional segregation and integration in the nervous system. 
        Proceedings of the National Academy of Sciences, 91(11), Article 11. 
        https://doi.org/10.1073/pnas.91.11.5033

    Parameters
    ----------
    X : np.ndarray[double, double]
        The multivariate time series in cells x time format.

    Returns
    -------
    np.ndarray[double]
        The instantanious local total correlation.

    """
    
    cdef DTYPE_int_t N0 = X.shape[0]
    cdef DTYPE_int_t N1 = X.shape[1]
    
    # The joint entropy of the whole
    cdef np.ndarray[DTYPE_double_t, ndim=1] joint_ents = local_entropy_nd(X)
    # Pre-allocating space for the sum of the local marginal entropies
    cdef np.ndarray[DTYPE_double_t, ndim=1] sum_marginal_ents = np.zeros(N1,
                                                                         dtype=DTYPE_double)
    cdef np.ndarray[DTYPE_double_t, ndim=1] marginal_ent 
    
    cdef DTYPE_int_t i     
    for i in range(N0):
        
        marginal_ent = local_entropy_1d(i, X)
        sum_marginal_ents = np.add(sum_marginal_ents, marginal_ent)
        
    return np.subtract(sum_marginal_ents, joint_ents)


@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def local_o_information(np.ndarray[DTYPE_double_t, ndim=2] X):
    """
    Returns a time series of the instantanious local O-information 
    for every frame in X. 
    
    If \Omega(X) < 0, the system is synergy-dominated.
    If \Omega(X) > 0, the system is redundancy-dominated. 
    
    O-information is:
        
        \omega(x) = (2-N)tc(x) + sum( tc(x^{-i}) )
    
    See: 
        Rosas, F., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019). 
        Quantifying High-order Interdependencies via Multivariate Extensions of the Mutual Information. 
        Physical Review E, 100(3), Article 3. 
        https://doi.org/10.1103/PhysRevE.100.032305
        
        Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023). 
        Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex. 
        Communications Biology, 6(1), Article 1. 
        https://doi.org/10.1038/s42003-023-04843-w
        
    Parameters
    ----------
    X : np.ndarray[double, double]
        The multivariate time series in cells x time format.

    Returns
    -------
    np.ndarray[double]
        The instantanious local o-informations

    """
    cdef DTYPE_double_t N0f = X.shape[0]
    cdef DTYPE_int_t N0 = X.shape[0]
    cdef DTYPE_int_t N1 = X.shape[1]
    cdef DTYPE_double_t factor = (2.0 - N0f)
    
    cdef np.ndarray whole_tc = np.multiply(factor,  
                                           local_total_correlation(X))
    
    # Pre-allocating the sum of the local residual TCs
    cdef np.ndarray[DTYPE_double_t, ndim=1] sum_residual_tcs = np.zeros(N1,
                                                                        dtype=DTYPE_double)
    
    # This will be the all X_j in X excluding X[i]
    cdef np.ndarray[DTYPE_double_t, ndim=2] X_residuals
    # The local TC series of X_residual
    cdef np.ndarray[DTYPE_double_t, ndim=1] residual_tc
    
    cdef DTYPE_int_t i, j
    for i in range(N0):
        
        X_residuals = X[[j for j in range(N0) if j != i],:]
        residual_tc = local_total_correlation(X_residuals)
        
        sum_residual_tcs = np.add(sum_residual_tcs, residual_tc)
    
    return np.add(whole_tc, sum_residual_tcs)


def local_dual_total_correlation(np.ndarray[DTYPE_double_t, ndim=2] X):
    """
    Returns a time series of the instantanious local dual total correlation 
    for every frame in X. 
    
    DTC(X) is low when all X_i are independnet
    DTC(X) is also low when all X_i are copies of each-other. 
    DTC(X) is high when there is higher-order information sharing. 
    
    Dual total correlation is:
        
        dtc(x) = h(x) - sum( h(x_i | x^{-i}) )
    
    See: 
        Rosas, F., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019). 
        Quantifying High-order Interdependencies via Multivariate Extensions of the Mutual Information. 
        Physical Review E, 100(3), Article 3. 
        https://doi.org/10.1103/PhysRevE.100.032305
        
        Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023). 
        Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex. 
        Communications Biology, 6(1), Article 1. 
        https://doi.org/10.1038/s42003-023-04843-w

    Parameters
    ----------
    X : np.ndarray[double, double]
        The multivariate time series in cells x time format.

    Returns
    -------
    np.ndarray[double]
        The instantanious local dual total correlation
        
    O = T - D
    O + D = T
    D = T - O
    """
    
    # Not the function described above, but the same.
    # See Varley et al., (2023)
    return np.subtract(local_total_correlation(X), local_o_information(X))


def local_s_information(np.ndarray[DTYPE_double_t, ndim=2] X):
    """
    Returns a time series of the instantanious S-information 
    for every frame in X. 
    
    DTC(X) is low when all X_i are independnet
    DTC(X) is high when both redundancy and synergy co-exist. 
    
    Dual total correlation is:
        
        \sigma(x) = \sum( i(x_i ; x^{-i}) )
    
    See: 
        Rosas, F., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019). 
        Quantifying High-order Interdependencies via Multivariate Extensions of the Mutual Information. 
        Physical Review E, 100(3), Article 3. 
        https://doi.org/10.1103/PhysRevE.100.032305
        
        Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023). 
        Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex. 
        Communications Biology, 6(1), Article 1. 
        https://doi.org/10.1038/s42003-023-04843-w

    Parameters
    ----------
    X : np.ndarray[double, double]
        The multivariate time series in cells x time format.

    Returns
    -------
    np.ndarray[double]
        The instantanious local s-information.
    """
    # Not the function described above, but this is the same. 
    # See Varley et al., (2023)
    return np.add(local_total_correlation(X), local_dual_total_correlation(X))


@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def local_tse_complexity(np.ndarray[DTYPE_double_t, ndim=2] X, 
                         DTYPE_int_t num_samples = 100):
    """
    Returns a time series of the instantanious Tononi-Sporns-Edelman complexity
    for every frame in X. 
    
    TSE(X) is low when all X_i are independnet
    TSE(X) is low when all X_i are copies of each-other
    TSE(X) is high when both integration and segregation co-exist. 
    
    TSE is:
        
        tse(x) = \sum( (k/N)*tc(x) - avg( tc(x^{k}) ) )
        
    Where avg( tc(x^{k})) is the average total correlation of all 
    subsets of x of size k. 
    
    See: 
        Tononi, G., Sporns, O., & Edelman, G. M. (1994). 
        A measure for brain complexity: Relating functional segregation and integration in the nervous system. 
        Proceedings of the National Academy of Sciences, 91(11), Article 11. 
        https://doi.org/10.1073/pnas.91.11.5033
        
        Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023). 
        Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex. 
        Communications Biology, 6(1), Article 1. 
        https://doi.org/10.1038/s42003-023-04843-w

    Parameters
    ----------
    X : np.ndarray[double, double]
        The multivariate time series in cells x time format.
    num_samples : int , optional
        The number of random subsets of each size k to sample. 
        The full number is too large to do. 
        The default is 100.
    Returns
    -------
    np.ndarray[double]
        The instantanious local tse_complexity.
    """
    cdef DTYPE_int_t N0 = X.shape[0]
    cdef DTYPE_double_t N0f = X.shape[0]
    cdef DTYPE_int_t N1 = X.shape[1]
    
    cdef np.ndarray[DTYPE_double_t, ndim=1] tc_whole = local_total_correlation(X)
    cdef np.ndarray[DTYPE_double_t, ndim=1] tse = np.zeros(N1,
                                                           dtype=DTYPE_double)
    
    cdef DTYPE_int_t k, i
    cdef DTYPE_double_t factor
    cdef np.ndarray[DTYPE_double_t, ndim=1] null_tc, sample_tcs_k
    cdef np.ndarray[DTYPE_double_t, ndim=2] X_choice
    cdef np.ndarray[DTYPE_int_t, ndim=1] choice 
    
    for k in range(1, N0):
        
        factor = float(k) / N0f
        null_tc = np.multiply(factor, tc_whole)
        
        sample_tcs_k = np.zeros(N1)
        
        for i in range(num_samples):
            if k > 1:
                
                choice = np.random.choice(N0, k, replace=False)
                X_choice = X[choice, :]
                sample_tcs_k = np.add(sample_tcs_k, local_total_correlation(X_choice))
        
        sample_tcs_k = np.divide(sample_tcs_k, float(num_samples))
        diff = np.subtract(null_tc, sample_tcs_k)
        
        tse = np.add(tse, diff)
        
    return tse
            


@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def average_total_correlation(np.ndarray[DTYPE_double_t, ndim=2] X, 
                              bint debug = False):
    """
    Computes the expected total correlation from X. 
    
    The output of this function should be very similar to
    local_total_correlation(X).mean()

    Parameters
    ----------
    X : np.ndarray 
        A multivariate time series object in cells x time format.

    Returns
    -------
    double
        The expected total correlation.

    """
    cdef np.ndarray[DTYPE_double_t, ndim=2] cov = np.cov(X, 
                                                         ddof=0.0)
    cdef DTYPE_int_t sign
    cdef DTYPE_double_t det
    
    # Using the logdet function since it is more powerfun 
    # then the default logdet function AND we'll be taking 
    # the logarithm of the determinant anyway.
    
    # The determinant of the matrix will always be positive 
    # since cov is symmetric. 
    sign, det = np.linalg.slogdet(cov)
        
    return -det / 2.0


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def average_o_information(np.ndarray[DTYPE_double_t, ndim=2] X):
    """
    Computes the expected O-information of X.
    
    The output of this function should be very similar to
    local_o_information(X).mean()
    
    Parameters
    ----------
    X : np.ndarray 
        A multivariate time series object in cells x time format.

    Returns
    -------
    double
        The average O-information.

    """
    cdef DTYPE_int_t N = X.shape[0]
    cdef DTYPE_double_t Nf = X.shape[0]
    cdef DTYPE_double_t factor = (2.-Nf)
    cdef DTYPE_double_t whole_tc = factor*average_total_correlation(X)
    
    cdef DTYPE_double_t sum_residual_tcs = 0.0
    
    cdef np.ndarray[DTYPE_double_t, ndim=2] X_residuals 
    cdef DTYPE_int_t i 
    
    for i in range(N):
        X_residuals = X[[j for j in range(N) if j != i],:]
        sum_residual_tcs += average_total_correlation(X_residuals)
    
    return whole_tc + sum_residual_tcs
    

def average_dual_total_correlation(np.ndarray[DTYPE_double_t, ndim=2] X):
    """
    
    The average dual total correlation.
    The output of this function should be very similar to
    local_dual_total_correlation(X).mean()
    
    Parameters
    ----------
    X : np.ndarray 
        A multivariate time series object in cells x time format.
    
    Returns
    -------
    double
    The average dual total correlation
    """
    
    return average_total_correlation(X) - average_o_information(X)


def average_s_information(np.ndarray[DTYPE_double_t, ndim=2] X):
    """
    The average S-information.
    The output of this function should be very similar to
    local_s_information(X).mean()
    
    Parameters
    ----------
    X : np.ndarray 
        A multivariate time series object in cells x time format.
    
    Returns
    -------
    double
    The average S-information
    """
    
    return average_total_correlation(X) + average_dual_total_correlation(X)


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def average_tse_complexity(np.ndarray[DTYPE_double_t, ndim=2] X, 
                           DTYPE_int_t num_samples = 100, 
                           bint return_plot = False):
    """
    Computes the expected TSE complexity from the Xset X.
    

    Parameters
    ----------
    X : np.ndarray 
        The multivariate time series in cells x time format.
    num_samples : int , optional
        The number of random subsets of each size k to sample. 
        The full number is too large to do. 
        The default is 100.
    return_plot : int YPE, optional
        If true, returns the TSE complexity, as well as the null TC and 
        average TC for each layer. 
        The default is False.

    Returns
    -------
    double
        The TSE complexity.
    np.ndarray
        A one-dimensional np.ndarray. 
        The average total correlation for nodes of size i
    np.ndarray 
        A one-dimensional np.ndarray.
        The expected null total correlation for nodes of size i. 
    """
    cdef DTYPE_int_t N = X.shape[0]
    cdef DTYPE_double_t Nf = X.shape[0]
    
    # Compute the total correlation of the whole.
    cdef DTYPE_double_t tc_whole = average_total_correlation(X)
    cdef DTYPE_double_t tse = 0.0
    
    # These are only used if return_plot == True
    cdef np.ndarray[DTYPE_double_t, ndim=1] null_tcs = np.zeros(N, 
                                                                dtype=DTYPE_double)
    cdef np.ndarray[DTYPE_double_t, ndim=1] sample_tcs = np.zeros(N,
                                                                  dtype=DTYPE_double)
    
    # Pre-allocating in-loop stuff.
    cdef DTYPE_double_t null_tc, sample_tcs_k 
    cdef np.ndarray[DTYPE_int_t, ndim=1] choice
    cdef np.ndarray[DTYPE_double_t, ndim=2] X_choice
    
    cdef DTYPE_int_t k, i
    for k in range(1,N): # Iterating over layers 
        null_tc = (float(k)/Nf)*tc_whole # The expected null TC. 
        
        if return_plot == True:
            null_tcs[k] = null_tc
        
        sample_tcs_k = 0.0 # This will become the average TC of size k
        for i in range(num_samples):
            if k > 1: # There's no point computing TC for single cells. 
            
                # Picking a subset of nodes of size k at random. 
                choice = np.random.choice(N, k, replace=False) # This is so slow.
                X_choice = X[choice, :] # Grabbing the subset from X. 
                sample_tcs_k += average_total_correlation(X_choice) # TC
        
        sample_tcs_k /= float(num_samples) # Make the sum into an average. 
        if return_plot == True:
            sample_tcs[k] = sample_tcs_k
        
        tse += (null_tc - sample_tcs_k)
    
    if return_plot == False:
        return tse
    else:
        return tse, sample_tcs, null_tcs

#%%

@cython.initializedcheck(False)
def local_phi_min(DTYPE_int_t idx1, DTYPE_int_t idx2, tuple atom, 
                  np.ndarray[DTYPE_double_t, ndim=2] X, 
                  DTYPE_int_t lag = 1):
    """
    Computes the local informative and misinformative probability mass exclusions.
   
   See:
       Finn, C., & Lizier, J. T. (2018). 
       Pointwise Partial Information Decomposition Using the Specificity and Ambiguity Lattices. 
       Entropy, 20(4), Article 4. 
       https://doi.org/10.3390/e20040297
       
       Mediano, P. A. M., et al., (2021). 
       Towards an extended taxonomy of information dynamics via 
       Integrated Information Decomposition. 
       arXiv:2109.13186 [Physics, q-Bio]. 
       http://arxiv.org/abs/2109.13186
       
       Luppi, A. I., et al., (2023). 
       A Synergistic Workspace for Human Consciousness Revealed by Integrated 
       Information Decomposition 
       (p. 2020.11.25.398081). bioRxiv. 
       https://doi.org/10.1101/2020.11.25.398081

    Parameters
    ----------
    x : int
        The index of the first process.
    y : int
        The index of the second process.
    atom : tuple
        The integrated information atom.
        A tuple of tuples of tuples (don't worry about it).
    X : np.ndarray
        The multidimensional time series in rows x time format.
    lag : int, optional
        The lag to compute the \PhiD w.r.t. The default is 1.

    Returns
    -------
    np.ndarray
        The local integrated information decomposition lattice.

    """
    cdef DTYPE_int_t N1 = X.shape[1]
    
    # This thing is called edge b/c of the earlier project I wrote it for. 
    cdef np.ndarray[DTYPE_double_t, ndim=2] edge = X[[idx1,idx2], :]
    
    cdef np.ndarray[DTYPE_double_t, ndim=1] i_plus = np.repeat(np.inf, N1-lag) # Informative 
    cdef np.ndarray[DTYPE_double_t, ndim=1] i_minus = np.repeat(np.inf, N1-lag) # Misinformative 
    
    cdef DTYPE_int_t i,j
    cdef DTYPE_int_t len_atom_0 = len(atom[0])
    cdef DTYPE_int_t len_atom_1 = len(atom[1])
    
    cdef np.ndarray h_edge_i, joint, marginal, conditional 
    
    for i in range(len_atom_0):
        
        # For i(s;t) = h(s) - h(s|t), h(s) is the informative probability mass exclusion. 
        # So the informative part of the MMI is just the entropy of the past process.
        edge_i = edge[((atom[0][i]),)][:,:-lag] # The time series of the ith element
        h_edge_i = local_entropy_nd(edge_i) # The entropy of edge i. 
        
        # Minimum entropy as redundancy. 
        i_plus = np.minimum(i_plus, 
                            h_edge_i
                            )
        
        # The misinformative probability mass exclusions: h(s|t)
        # We are taking the min over all s and t. 
        # h(s|t) = h(s,t) - h(t) 
        for j in range(len_atom_1):
            
            # Joint-state of past and future.
            joint = np.squeeze(
                        np.vstack((
                            edge[((atom[0][i],))][:,:-lag], 
                            edge[((atom[1][j],))][:,lag:]
                            ))
                        )
            # Marginal future
            marginal = edge[((atom[1][j],))][:,lag:]
            # h(s|t) = h(s,t) - h(t)
            conditional = np.subtract(local_entropy_nd(joint),
                                      local_entropy_nd(marginal)
                                      )
            # Minimum redundancy
            i_minus = np.minimum(i_minus,
                                 conditional
                )
    # mi is informative minus misinformative. 
    return np.subtract(i_plus, i_minus)


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def local_phi_id(int idx1, int idx2, np.ndarray X, int lag = 1):
    """
    Computes the local integrated information decomposition for two variables
    X[x] and X[y] in X. 
    
    Returns a sixteen-node networkx DiGraph object, where each node has an attribute
    "pi" that contians the information time series for that atom. 

    Parameters
    ----------
    idx1 : int
        The index of the first process.
    idx2 : int
        The index of the second process.
    X : np.ndarray
        The multidimensional time series in rows x time format.
    lag : int, optional
        The lag to compute the \PhiD w.r.t. The default is 1.

    Returns
    -------
    lattice : networkx DiGraph
        The integrated information lattice.

    """
    lattice = deepcopy(lattice_orig)
    
    cdef tuple atom

    # Going up the lattice in ascending order
    for atom in order:
        # Compute the redundancy
        lattice.nodes[atom]["phi_min"] = local_phi_min(idx1, idx2, 
                                                       atom, X)
        
        # At the bottom, the redundancy is the partial information
        if atom == (((0,),(1,)),((0,),(1,))):
            lattice.nodes[atom]["pi"] = lattice.nodes[atom]["phi_min"]
        else: # Mobius inversion
            lattice.nodes[atom]["pi"] = np.subtract(lattice.nodes[atom]["phi_min"],
                                                    np.vstack(([lattice.nodes[a]["pi"] for a in lattice.nodes[atom]["descendants"]])).sum(axis=0)
                                                    )
    return lattice


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def local_phi_r(phi_lattice):
    """
    A non-negative derivaive of \Phi^{WMS}. This term adds back in the 
    dynamic redundancy, which is double-subtracted in the original 
    computation. 
    
    See:
        Mediano, P. A. M., et al., (2021). 
        Towards an extended taxonomy of information dynamics via Integrated Information Decomposition. 
        arXiv:2109.13186 [Physics, q-Bio]. 
        http://arxiv.org/abs/2109.13186

    Parameters
    ----------
    phi_lattice : nx.DiGraph()
        The integrated information lattice.

    Returns
    -------
    phir : np.ndarray
        The local corrected \Phi^{WMS} for each frame. 

    """
    
    # Phir is the sum of a subset of integrated information atoms
    phir = phi_lattice.nodes[(((0,),), ((0, 1),))]["pi"]
    
    for atom in phir_atoms:
        phir = phi_lattice.nodes[atom]["pi"]
    
    return phir
#%%

@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def global_signal_regression(np.ndarray[DTYPE_double_t, ndim=2] X):
    """
    Regresses out the global signal (mean of the time seriese) from 
    each of the channels.

    Parameters
    ----------
    series : np.ndarray
        The raw time series in channels x time format.

    Returns
    -------
    gsr : np.ndarray
        The transformed time series after GSR has been applied.

    """
        
    cdef DTYPE_int_t N0 = X.shape[0] # Number of rows (channels)
    cdef DTYPE_int_t N1 = X.shape[1] # The number of columns (frames)
        
    cdef np.ndarray[DTYPE_double_t, ndim=2] gsr = np.zeros((N0, N1), 
                                                           dtype=DTYPE_double) # Initialize GSR array
    cdef np.ndarray[DTYPE_double_t, ndim=1] mean = np.mean(X, axis=0) # Compute global signal
            
    cdef DTYPE_int_t i, j
    cdef np.ndarray[DTYPE_double_t, ndim=1] ypred, z
    for i in range(N0): 
        lr = linregress(mean, X[i]) # Linregress each channel against the GS
        ypred = lr[1] + (lr[0]*mean)
        
        z = np.subtract(X[i], ypred) # Regress out
        for j in range(N1): # No need to iterate over columns, but it's fine. 
            gsr[i,j] = z[j] # From an earlier function in C. 
        
    return zscore(gsr, axis=-1)


@cython.initializedcheck(False)
@cython.boundscheck(False)
def remove_autocorrelation(np.ndarray[DTYPE_double_t, ndim=2] X):
    """
    Removes the lag-1 autocorrelation.

    Parameters
    ----------
    series : np.ndarray
        The raw time series in channels x time format.

    Returns
    -------
    np.ndarray
        The X with the univariate autocorrelation regressed out.

    """
    cdef DTYPE_int_t N0 = X.shape[0]
    cdef DTYPE_int_t N1 = X.shape[1]
    
    regressed = np.zeros((N0, N1-1))
    
    cdef DTYPE_int_t i 
    cdef np.ndarray[DTYPE_double_t, ndim=1] X_i
    
    for i in range(N0): # Each row is regressed independently of all others. 
        
        X_i = X[i]
        
        # Computing the linear correlation between time {t-1} and time {t}
        lr = linregress(X_i[:-1], X_i[1:])
        # The predicted values at time {t} given the regression. 
        ypred = lr[1] + (lr[0]*X_i[:-1])
        # Computing the residuals. 
        residuals = np.subtract(X_i[1:], ypred)
        
        regressed[i,:] = residuals
    
    return zscore(regressed, axis=-1)


@cython.initializedcheck(False)
def mutual_information_matrix(np.ndarray[DTYPE_double_t, ndim=2] X,
                              DTYPE_double_t alpha,
                              DTYPE_int_t lag = 1,
                              bint bonferonni = True):
    """
    Computes the symmetric matrix of binary mutual informations
    between each element of the X array. 
    
    Since we care about information flow, it makes sense to offset the 
    time series by some variable {lag}. In the matrix, cell 
    ij gives the toal information flow from i -> j + j -> i. 

    Parameters
    ----------
    series : np.ndarray
        The raw time series in channels x time format..
    alpha : double
        The significance level.
    lag : int, optional 
        The lag used to account for time. 
        The default is 1. 
    bonferonni : bool, optional
        Whether to use a bonferonni correction. 
        The default is True.

    Returns
    -------
    np.ndarray
        The bivariate mutual information matrix.

    """
        
    cdef DTYPE_int_t N0 = X.shape[0]
    cdef DTYPE_double_t N0f = X.shape[0]
    cdef DTYPE_double_t[:,:] mimat = np.zeros((N0, N0))
    
    # The bonferonni correction. 
    cdef DTYPE_double_t alpha_corr
    
    if bonferonni == True:
        alpha_corr = alpha / (((N0f**2.0)-N0f) / 2.0)
    else:
        alpha_corr = 1*alpha
    
    cdef DTYPE_int_t i, j
    cdef DTYPE_double_t r, r1, r2, p, p1, p2, mi, mi1, mi2
    for i in range(N0):
        for j in range(i): # You only need to do the upper triangle. 
            
            if lag == 0:
                r, p = pearsonr(X[i], X[j])
                if p < alpha_corr: # No point computing a log if you don't have to...
                    mi = -0.5*log(1-(r**2.0)) # Gaussian MI from Pearson's r.
                
                    mimat[i,j] = mi
                    mimat[j,i] = mi
                    
            elif lag > 0:
                
                r1, p1 = pearsonr(X[i,:-lag], X[j,lag:])
                r2, p2 = pearsonr(X[i,lag:], X[j,:-lag])
                
                if p1 < alpha_corr:
                    mi1 = -0.5*log(1.0-(r1**2.0))
                else:
                    mi1 = 0
                
                if p2 < alpha_corr:
                    mi2 = -0.5*log(1.0-(r2**2.0))
                else:
                    mi2 = 0
                
                mimat[i,j] = mi1 + mi2 
                mimat[j,i] = mi1 + mi2
                
    return np.array(mimat)

@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def minimum_information_bipartition(np.ndarray[DTYPE_double_t, ndim=2] mimat, 
                                    bint noise=False,
                                    DTYPE_double_t noise_level = 10**-6): 
    """
    Computes the minimum information bipartition of a 

    Parameters
    ----------
    mimat : np.ndarray[double, double]
        A symmetric mutual information matrix.
    noise : bool 
        Whether to add noise to the graph to ensure it is connected. 
    noise_level : double
        The amount of noise to add across the matrix. 

    Returns
    -------
    bipartition : list
        A list of two lists, each sub-list contains the indices 
        of the nodes in the associated bipartition.
    """
    
    cdef DTYPE_int_t N0 = mimat.shape[0]
    
    # The noise is a uniform floor of low mutual information connections. 
    # This will ensure the graph is connected, but will slightly weaken 
    # the strength of the modular struture. 
    if noise == True: 
        mimat_corr = np.add(mimat, noise_level)
    else: 
        mimat_corr = 1*mimat
    
    # Constructing the networkx object. 
    G = nx.from_numpy_array(mimat_corr, create_using=nx.Graph())
    
    # Setting normalized to True increases the runtime dramatically.
    # I don't really know why.
    fiedler = nx.fiedler_vector(G,
                                weight="weight",
                                normalized=False)

    cdef DTYPE_int_t i 
    cdef list bipartition = [
        [i for i in range(N0) if fiedler[i] > 0],
        [i for i in range(N0) if fiedler[i] < 0]
        ]

    return bipartition


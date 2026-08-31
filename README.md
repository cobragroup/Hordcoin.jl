# Hordcoin.jl

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/cobragroup/Hordcoin.jl/blob/main/LICENSE)

Hordcoin provides methods for finding probability distributions with maximal Shannon entropy given a fixed marginal distribution or entropy up to a chosen order, and to compute the Connected Information. The package allows the selection of different optimisers.

This project was created as a part of the bachelor's thesis "Connected Information from Given Entropies" at the Faculty of Electrical Engineering, Czech Technical University in Prague, and of the paper "HORDCOIN: A Software Library for Higher Order Connected Information and Entropic Constraints Approximation". See the section [How to cite](#how-to-cite) to cite it appropriately.

To maximise the entropy with marginal constraints, the package implements the following methods:
- Exponential Cone Programming (with different solvers)
- Iterative Proportional Fitting Procedure
- Projected Gradient Descent

To maximize entropy while satisfying entropic constraints, the package employs a polymatroid approximation; refer to the paper for details.
Moreover, in case of undersampled distributions, it is possible to use a built-in small sample correction for the values of entropy instead of the plug-in estimator.


## Installation
The package is registered, it can be installed by simply:
```julia
pkg> add Hordcoin
```

## Usage
The primary functionality of this package is to implement methods that maximize the Shannon entropy of a probability distribution with marginal distribution or entropic constraints and compute the Connected Information.

The input data must satisfy the following requirements:
- The probability distributions are stored as multidimensional arrays;
- Probabilities are non-negative and sum up to 1;
- OR are provided as (non-normalised) counts;
- The maximal order of the fixed marginal distributions has to be in [2, n-1], where n is the number of dimensions of the probability distribution.

### Connected Information

The main function of the package is `connected_information` that uses the the maximum entropy with constraints at different orders to compute the Connected Information. It takes as input the probability distribution or the (non-normalised) counts, along with the desired orders of Connected Information and the optimisation method.

When computing multiple Connected Information values for the same probability distribution, it is possible to pass the sizes (desired orders) as an array. This will speed up the process by chaining the computations and caching intermediate results, thereby reducing the number of maximizations.

If no method is passed, the kind of optimisation is decided by the data type of the input:
- Int input triggers constraints on the marginal entropy and the RawPolymatroid method;
- Float input triggers constraints on the marginal distributions and the Ipfp method.

It is possible to have complete control on the kind of constraints by passing a method explicitly:
- Gradient, Cone and Ipfp trigger constraints on the marginal distributions with both Int and Float inputs;
- RawPolymatroid and GPolymatroid require constraints on the marginal entropy. However, GPolymatroid will raise an error if used with Float inputs as it needs the counts to compute the correction. When using Polymatroid methods, it's possible to pass `precalculated_entropies` as a keyword parameter. This is useful if the entropies are cached elsewhere or if their estimate has a different origin (perhaps a continuous distribution).


The basic usage of `connected_information` is the following:
```julia
using Hordcoin

counts=cat([1 2; 3 4], [4 2; 1 3], dims=3);
connected_information(counts, 2)
```
Which will optimise (maximize entropy) constraining the marginal entropies (up to order 2) and should give a result similar to `(Dict(2 => 0.09310598013744764),Dict{Int64, EResult}(2 => EMFMEResult(2.89221815884257, Dict{Vector{Int64}, Float64}()), 1 => EMFMEResult(2.985324138980082, Dict{Vector{Int64}, Float64}()))`

The first dictionary contains the values of CI at the requested orders, the second contains the entropy of the maximally entropic distribution for every order of the constraints and an empty dictionary. Calling `connected_information` with the keyword argument `full_output=true` populated that dictionary with the entropy of all the marginal distributions. In this example something like `Dict{Vector{Int64}, Real}([2, 3] => 1.9854752967410236, [3] => 1.0000000007841423, [] => 2.0528470297191336e-11, [1] => 0.9927744539824459, [1, 3] => 1.9261207462548557, [1, 2, 3] => 2.8922181588425695, ...)`. This output can be large for high dimensional distributions, but can be useful for further analysis. It can be obtained from `maximise_entropy` (see below). In case of optimisation fixing the marginal distributions, the second element is an array containing the optimised probability distribution.

Notably, the following operations all give the same results:
```julia
connected_information(counts, [2])
connected_information(counts, 2, RawPolymatroid())
frequencies = counts ./ sum(counts);
connected_information(frequencies, 2, RawPolymatroid())
```

Alternatively, it's possible to trigger the marginal distribution constraints with these equivalent lines:
```julia
connected_information(frequencies, 2)
connected_information(counts, 2, Ipfp())
connected_information(frequencies, [2], Ipfp())
```

Or similar results with:
```julia
connected_information(frequencies, 2, Gradient())
connected_information(frequencies, 2, Cone())
connected_information(frequencies, 2, Cone(SCS.Optimizer()))
connected_information(frequencies, 2, Cone(MosekTools.Optimizer()))
```
Where the last one requires a Mosek license. (Academic licence easy to obtain at https://www.mosek.com/products/academic-licenses/).

Other useful parameters for the Polymatroid methods are:
- zhang_yeung: to enable the Zhang-Yeung inequalities complementing the Shannon inequalities and improving the approximation at higher orders (see paper),
- optimizer: to chose between the SCS and the Mosek optimiser
- mle_correction: (only RawPolymatroid) enables a rough correction for the finite sample
- tolerance: (only GPolymatroid) enables a relaxation of the constraints to help convergence (sometimes it fails with the corrected entropies). Note: CI estimate can become negative due to the relaxed constraints.

### Other functions

It is possible to access directly the entropy maximisation through the functions `maximise_entropy` for marginal constraints and entropic constraints. `maximise_entropy` takes as an input a probability distribution and the order of marginal distributions to constrain. The optimiser is an optional parameter that can have further specified parameters (such as the number of iterations, etc.). The function returns the probability distribution with maximal entropy in the form of `EMResult` (Entropy Maximisation Result) or an `EMFMEResult` (Entropy Maximisation with Fixed Marginal Entropies Result) depending of the method chosen. It's possible to pass a precomputed dictionary of entropies to speed up the computation.

The basic usage is the following:
```julia
using Hordcoin

probability_distribution = [1/16; 3/16;; 3/16; 1/16;;; 1/16; 3/16;; 3/16; 1/16]
marginal_size = 2
maximise_entropy(probability_distribution, marginal_size)
```
Running the code with the optional parameter `method`:
```julia
using SCS

maximise_entropy(probability_distribution, marginal_size, method = Gradient(10, SCS.Optimizer()))
```

The package also contains three utility functions. `distribution_entropy` computes the information entropy of a probability distribution. `permutations_of_length` returns all permutations of a given size from elements from 1 to dims. `precompute_entropies` computes the entropy of all the marginal distributions to use with polymatroid methods choosing the plug-in estimator (possibly with MLE correction) or Grassberger correction.

Usage of the functions:
```julia
distribution_entropy(probability_distribution)
permutations_of_length(3, 4)
precompute_entropies(probability_distribution)
```



## Recommendations

The most efficient method when computing with fixed marginal distributions is the `Cone` method with `MosekTools.Optimizer()`. This requires a license to use the MOSEK solver. Without the license, it is possible to use `SCS.Optimizer()` instead, but it is less accurate and slower.

Without a MOSEK license, use the `Ipfp` method (default). It is accurate and not the slowest. It can also be parametrized with the number of iterations, but it is not necessary. The default value is 10.

The `Gradient` method is the slowest and may fail during execution due to limitations of Second Order Cone constraints in solvers.

When computing with fixed entropies and a small number of samples, the recommended method is the `GPolymatroid` with `MosekTools.Optimizer()`. When the distribution is sampled enough, you can use `RawPolymatroid` to estimate the entropy with the plug-in estimator. More information can be found in the paper.

## How to cite

If you use this code for a scientific publication, please cite:

> Tani Raffaelli G., Kislinger J., Kroupa T., and Hlinka J., "HORDCOIN: A Software Library for Higher Order Connected Information and Entropic Constraints Approximation"

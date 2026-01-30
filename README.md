# HORDCOIN.jl

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/cobragroup/HORDCOIN/blob/main/LICENSE)

HORDCOIN provides methods for finding probability distributions with maximal Shannon entropy given a fixed marginal distribution or entropy up to a chosen order, and to compute the Connected Information. The package allows the selection of different optimisers.

This project was created as a part of the bachelor's thesis "Connected Information from Given Entropies" at the Faculty of Electrical Engineering, Czech Technical University in Prague, and of the paper "HORDCOIN: A Software Library for Higher Order Connected Information and Entropic Constraints Approximation". See the section [How to cite](#how-to-cite) to cite it appropriately.

To maximise the entropy with marginal constraints, the package implements the following methods:
- Exponential Cone Programming (with different solvers)
- Iterative Proportional Fitting Procedure
- Projected Gradient Descent

To maximize entropy while satisfying entropic constraints, the package employs a polymatroid approximation; refer to the paper for details.
Moreover, in case of undersampled distributions, it is possible to use a built-in small sample correction for the values of entropy instead of the plug-in estimator.


## Installation
The package is not registered, but can be installed in the following way:
```julia
pkg> add https://github.com/cobragroup/HORDCOIN.git#develop
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

When computing multiple Connected Information values for the same probability distribution, it is possible to pass the sizes (desired orders) as an array. This will speed up the process by chaining the computations, thereby reducing the number of maximizations.

If no method is passed, the kind of optimisation is decided by the data type of the input:
- Int input triggers constraints on the marginal entropy and the RawPolymatroid method;
- Float input triggers constraints on the marginal distributions and the Ipfp method.

It is possible to have complete control on the kind of constraints by passing a method explicitly:
- Gradient, Cone and Ipfp trigger constraints on the marginal distributions with both Int and Float inputs;
- RawPolymatroid and GPolymatroid require constraints on the marginal entropy. However, GPolymatroid will raise an error if used with Float inputs as it needs the counts to compute the correction.


The basic usage of `connected_information` is the following:
```julia
using HORDCOIN

counts=cat([1 2; 3 4], [4 2; 1 3], dims=3);
connected_information(counts, 2)
```
Which will optimise (maximize entropy) constraining the marginal entropies (up to order 2) and should give a result similar to `(Dict(2 => 0.09310598013744764), Dict(2 => 2.892218158842619, 1 => 2.9853241389800664))`

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
- tolerance: (only GPolymatroid) enables a relaxation of the constraints to help convergence (sometimes if fails with the corrected entropies). Note: CI estimate can become negative due to the relaxed constraints.

### Other functions

It is possible to access directly the entropy maximisation through the functions `maximise_entropy` for marginal constraints and its sibling function `max_ent_fixed_ent_unnormalized` for the entropic constraints. `maximise_entropy` takes as an input a probability distribution and the order of marginal distributions to constrain. The optimiser is an optional parameter that can have further specified parameters (such as the number of iterations, etc.). The function returns the probability distribution with maximal entropy in the form of `EMResult`.
`max_ent_fixed_ent_unnormalized` takes a multidimensional array of counts and the order up to which the marginal entropies must be fixed. It allows the selection of the plug-in estimator for the entropies or the corrected one. It's possible to pass a precomputed dictionary of entropies to speed up the computation.

The basic usage is the following:
```julia
using HORDCOIN

probability_distribution = [1/16; 3/16;; 3/16; 1/16;;; 1/16; 3/16;; 3/16; 1/16]
marginal_size = 2
maximise_entropy(probability_distribution, marginal_size)
```
Running the code with the optional parameter `method`:
```julia
using SCS

maximise_entropy(probability_distribution, marginal_size; method = Gradient(10, SCS.Optimizer()))
```

The package also contains two utility functions. `distribution_entropy` computes the information entropy of a probability distribution. `permutations_of_length` returns all permutations of a given size from elements from 1 to dims. 

Usage of the functions:
```julia
distribution_entropy(probability_distribution)
permutations_of_length(3, 4)
```



## Recommendations

The most efficient method when computing with fixed marginal distributions is the `Cone` method with `MosekTools.Optimizer()`. This requires a license to use the MOSEK solver. Without the license, it is possible to use `SCS.Optimizer()` instead, but it is less accurate and slower.

Without a MOSEK license, use the `Ipfp` method (default). It is accurate and not the slowest. It can also be parametrized with the number of iterations, but it is not necessary. The default value is 10.

The `Gradient` method is the slowest and may fail during execution due to limitations of Second Order Cone constraints in solvers.

When computing with fixed entropies and a small number of samples, the recommended method is the `GPolymatroid` with `MosekTools.Optimizer()`. When the distribution is sampled enough, you can use `RawPolymatroid` to estimate the entropy with the plug-in estimator. More information can be found in the paper.

## How to cite

If you use this code for a scientific publication, please cite:

> Tani Raffaelli G., Kislinger J., Kroupa T., and Hlinka J., "HORDCOIN: A Software Library for Higher Order Connected Information and Entropic Constraints Approximation"

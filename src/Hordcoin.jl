# Hordcoin.jl:

module Hordcoin

using ProgressMeter

using Combinatorics

using SCS
using MathOptInterface
using JuMP
using MosekTools
using Ipopt
using MadNLP

using CDDLib

export Cone
export Gradient
export Ipfp

export Direct
export PolymatroidEntropyMethod
export RawPolymatroid
export GPolymatroid

export maximise_entropy
export connected_information

export distribution_entropy
export permutations_of_length
export precompute_entropies

export EResult
export EMResult
export EMFMEResult

include("types.jl")
include("utils.jl")
include("Gcorr.jl")
include("ipfp.jl")
include("exponentialCone.jl")
include("projectedGradient.jl")
include("polymatroid.jl")


"""
	maximise_entropy(joint_probability::Array{<:Real}, marginal_size::S, method::AbstractMaximizationMethod) -> EMResult where S<:Integer

Find the maximum-entropy distribution whose marginals of size `marginal_size` match those of `joint_probability`.

# Arguments
- `joint_probability::Array{<:Real}`: N-dimensional probability table (array) whose elements (non-negative) sum to ≈ 1 or an N-dimensional table of counts.
- `marginal_size::Int`: Order of the marginals of the constraint.
- `method::AbstractMaximizationMethod`: The method used for the maximisation.

# Returns
- `EResult`: A result object holding the maximum **entropy** and more information about the maximally entropic distribution.
"""
function maximise_entropy end

"""
	maximise_entropy(joint_probability::Array{<:AbstractFloat}, marginal_size::S, method::AbstractMarginalMethod) where S<:Integer -> EMResult

When called with a MarginalMethod fixes **all** marginals of order `marginal_size` and maximizes Shannon entropy over the feasible set.

# Arguments
- `marginal_size::Int`: Order of the marginals to hold fixed. For example, `2` fixes every pairwise marginal.
- `method::AbstractMarginalMethod = Cone()`:
	- `Cone([optimizer])`: entropy maximization via exponential cone programming.
	- `Gradient(; iterations, optimiser)`: projected-gradient approach.
	- `Ipfp(; iterations)`: iterative proportional fitting (IPFP).

# Returns
- `EResult`: A result object holding the maximum **entropy** and the **maximally entropic distribution**

# Throws
- `DomainError` if `marginal_size > ndims(joint_probability)`.
- `DomainError` if `marginal_size < 1`.
- `DomainError` if `sum(joint_probability)` is not approximately `1`.

If `marginal_size == ndims(joint_probability)`, the input is already fully specified; the function returns it unchanged.

# Examples
```julia-repl
julia> x = [0.1 0.4; 0.4 0.1]
2×2 Matrix{Float64}:
	0.1  0.4
	0.4  0.1

julia> maximise_entropy(x, 2)
Entropy: 1.7219280948873623
Distribution:
[0.1 0.4; 0.4 0.1]

julia> maximise_entropy(x, 1; method = Ipfp())
Progress: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:00
Entropy: 2.0
Distribution:
[0.25 0.25; 0.25 0.25]
```
"""
function maximise_entropy(joint_probability::Array{<:AbstractFloat}, marginal_size::S, method::AbstractMarginalMethod)::EMResult where S<:Integer

	marginal_size > ndims(joint_probability) &&
		throw(DomainError("Marginal size cannot be greater than number of dimensions of joint probability"))
	marginal_size < 1 &&
		throw(DomainError("Marginal size has to be positive"))
	!isapprox(sum(joint_probability), 1) &&
		throw(DomainError("Joint probability has to sum to 1"))

	if marginal_size == ndims(joint_probability)
		return EMResult(joint_probability)
	end

	marginals = permutations_of_length(marginal_size, ndims(joint_probability))

	return maximise_method(joint_probability, marginals, method)
end

"""
	maximise_entropy(joint_probability::Array{<:AbstractFloat}, marginal_size::S) where S<:Integer -> EMResult

When called with an array of floats assume fixed marginal distributions and defaults to the Ipfp method.
"""
function maximise_entropy(joint_probability::Array{<:AbstractFloat}, marginal_size::S)::EMResult where S<:Integer
	return maximise_entropy(joint_probability, marginal_size, Ipfp())
end

"""
	maximise_entropy(counts::Array{<:Integer}, marginal_size::S, method::AbstractMarginalMethod) where S<:Integer -> EMResult

When called with an array of integer counts and requiring maximisation with fixed marginal distributions estimate the probability with the frequency.
"""
function maximise_entropy(counts::Array{<:Integer}, marginal_size::S, method::AbstractMarginalMethod)::EMResult where S<:Integer
	return maximise_entropy(counts./sum(counts), marginal_size, method)
end

function maximise_method(joint_probability::Array{<:AbstractFloat}, marginals, method::Cone)
	cone_for_optimiser(joint_probability, marginals, method.optimiser)
end

function maximise_method(joint_probability::Array{<:AbstractFloat}, marginals, method::Gradient)
	descent(joint_probability, marginals; iterations = method.iterations, optimiser = method.optimiser)
end

function maximise_method(joint_probability::Array{<:AbstractFloat}, marginals, method::Ipfp)
	ipfp(joint_probability, marginals, iterations = method.iterations)
end

"""
	connected_information(joint_probability::Array, orders, method; precalculated_entropies, full_output::Bool=false) -> Tuple{Dict{Int, Float64}, Dict{Int, EResult}}

Compute connected information for **multiple orders** efficiently evaluating the maximum entropy for constraints at orders `m` and `m-1` for each requested order `m`.

# Arguments
- `joint_probability::Array{<:Real}`: N-dimensional probability table (array) whose elements (non-negative) sum to ≈ 1 or an N-dimensional table of counts.
- `orders`: Integer or vector of Integers. Order of the CI to compute. Values must satisfy `2 ≤ orders[i] ≤ ndims(joint_probability)`.
- `method::AbstractMaximizationMethod`: The method used for the maximisation.

# Keywords
- `precalculated_entropies`: A dictionary of precalculated entropies used only by AbstractEntropyMethods.

# Returns
- `Tuple{Dict{Int, Float64}, Dict{Int, Float64}}`: A tuple containing two dictionaries: the first one maps order to connected information and the second one maps marginal size to entropy for the orders actually computed.
"""

function connected_information end

"""
	connected_information(joint_probability::Array{<:Real}, orders::Vector{Int}, method = Ipfp(); precalculated_entropies, full_output::Bool=false) -> Tuple{Dict{Int, Float64}, Dict{Int, EMResult}}

When called with a MarginalMethod fixes **all** marginals of order `marginal_size` and maximizes Shannon entropy over the feasible set.

# Arguments
- `joint_probability::Array{<:Real}`: N-dimensional probability table summing to ~1.
- `method = Ipfp()`: Optimisation strategy used inside repeated `maximise_entropy` calls. Can be one of:
	- `Cone(optimiser = SCS.Optimizer())`: cone programming.
	- `Gradient(iterations, optimiser = SCS.Optimizer())`: gradient-based approach, default number of iterations is 10.
	- `Ipfp([iterations])`: Iterative proportional fitting, default number of iterations is 10.

# Keywords
- `precalculated_entropies`: Ignored.

# Throws
- `DomainError` if any `orders[i] > ndims(joint_probability)` or if any `orders[i] < 2`.

# Example
```julia-repl
julia> x = [0.25; 0;; 0; 0.25;;; 0; 0.25;; 0.25; 0]
2×2×2 Array{Float64, 3}:
[:, :, 1] =
 0.25  0.0
 0.0   0.25

[:, :, 2] =
 0.0   0.25
 0.25  0.0

julia> connected_information(x, [2, 3]; method = Ipfp())
Dict{Int64, Float64} with 2 entries:
  2 => 0.0
  3 => 1.0
```
"""
function connected_information(joint_probability::Array{T}, orders::Vector{<:Integer}, method::AbstractMarginalMethod; precalculated_entropies = Dict{Vector{Int}, Real}(), full_output::Bool=false)::Tuple{Dict{Int, Float64}, Dict{Int, EMResult}} where T <: Real

	sort!(orders)

	max_size = orders[end]
	min_size = orders[1]

	max_size > ndims(joint_probability) &&
		throw(DomainError("Marginal size cannot be greater than number of dimensions of joint probability"))
	min_size < 2 &&
		throw(DomainError("Marginal size for connected information cannot be less than 2"))

	set_marginals = Set([orders..., (orders .- 1)...])
	dict_results = Dict{Int, EMResult}()

	for m in set_marginals
		emresult = maximise_entropy(joint_probability, m, method)
		if full_output
			dict_results[m] = emresult
		else
			dict_results[m] = EMResult(emresult.entropy)
		end
	end

	ret_dict = Dict{Int, Float64}()

	for m in orders
		ret_dict[m] = dict_results[m-1].entropy - dict_results[m].entropy
	end

	return ret_dict, dict_results
end

"""
	connected_information(joint_probability::Array{T}, orders::Vector{<:Integer}; precalculated_entropies = Dict{Vector{Int}, Real}()) where T <: AbstractFloat -> Tuple{Dict{Int, Float64}, Dict{Int, Float64}}

When called with an array of floats assume fixed marginal distributions and defaults to the Ipfp method.
"""
function connected_information(joint_probability::Array{T}, orders::Vector{<:Integer}; precalculated_entropies = Dict{Vector{Int}, Real}(), full_output::Bool=false)::Tuple{Dict{Int, Float64}, Dict{Int, EMResult}} where T <: AbstractFloat
	return connected_information(joint_probability, orders, Ipfp(); full_output)
end

"""
	onnected_information(counts::Array{<:Integer}, orders::Vector{<:Integer}, method::AbstractMarginalMethod; precalculated_entropies = Dict{Vector{Int}, Real}()) -> Tuple{Dict{Int, Float64}, Dict{Int, Float64}}

When called with an array of integer counts and requiring maximisation with fixed marginal distributions estimate the probability with the frequency.
"""
function connected_information(counts::Array{<:Integer}, orders::Vector{<:Integer}, method::AbstractMarginalMethod; precalculated_entropies = Dict{Vector{Int}, Real}(), full_output::Bool=false)::Tuple{Dict{Int, Float64}, Dict{Int, EMResult}}
	connected_information(counts ./ sum(counts), orders, method; full_output)
end

"""
	maximise_entropy(joint_probability::Array{<:Real}, marginal_size::S, method::AbstractEntropyMethod; precalculated_entropies = Dict{Vector{Integer}, Real}()) where S<:Integer -> EResult

When called with an EntropyMethod fixes the marginal distribution entropies for all marginals up to order `marginal_size` and maximises the probability of the distribution.

# Arguments
- `marginal_size::Int`: Order of marginals up to which the **entropy** is kept fixed. For example, `2` fixes the entropy of 1D and 2D marginals.
- `method::AbstractEntropyMethod = RawPolymatroid()`:
	- `Direct([optimiser])`: Non-Linear Programming optimisation. The optimiser can be either "ipopt" or "madnlp".
	- `RawPolymatroid([mle_correction, zhang_yeung, optimiser])`: Polymatroid optimisation with naive estimate of marginal entropies. Optionally include MLE correction and Zhang–Yeung inequalities. The optimiser can be an instance of "SCS" or "Mosek".
	- `GPolymatroid([zhang_yeung, optimiser, tolerance])`: Polymatroid optimisation with Grassberger entropy estimator. Optionally include Zhang–Yeung inequalities. The optimiser can be an instance of "SCS" or "Mosek". The tolerance is a relaxation of the constraints to help convergence, can lead to negative CI.

# Throws
- `DomainError` if `marginal_size > ndims(joint_probability)` or `marginal_size < 1`.
- `DomainError` if `sum(joint_probability)` is not approximately `1` when using floats.
"""
function maximise_entropy(
	joint_probability::Array{<:Real},
	marginal_size::S,
	method::AbstractEntropyMethod;
	precalculated_entropies = Dict{Vector{Integer}, Real}())::EResult where S<:Integer
	marginal_size > ndims(joint_probability) &&
		throw(DomainError("Marginal size cannot be greater than number of dimensions of joint probability"))
	marginal_size < 1 &&
		throw(DomainError("Marginal size has to be positive"))

	if joint_probability isa Array{<:AbstractFloat}
		if ! ((method isa RawPolymatroid) || (method isa Direct))
			throw(DomainError("Method must be either RawPolymatroid or Direct for fixed entropy maximisation and normalized distribution."))
		end # better failing soon with a better error message
		!isapprox(sum(joint_probability), 1) &&
			throw(DomainError("Joint probability has to sum to 1"))
	end

	return _max_ent(joint_probability, marginal_size, method; precalculated_entropies)
end

"""
	maximise_entropy(counts::Array{<:Integer}, marginal_size::S; precalculated_entropies = Dict{Vector{Integer}, Real}()) where S<:Integer -> EMFMEResult

When called with an array of integers assume fixed marginal distributions entropies and defaults to the RawPolymatroid method.
"""
function maximise_entropy(counts::Array{<:Integer}, marginal_size::S; precalculated_entropies = Dict{Vector{Integer}, Real}())::EMFMEResult where S<:Integer
	return maximise_entropy(counts, marginal_size, RawPolymatroid(); precalculated_entropies = precalculated_entropies)
end

function _max_ent(joint_probability::Array{<:AbstractFloat}, marginal_size::S, method::Direct; precalculated_entropies = Dict())::EMResult where S<:Integer
	return nlp_entropies_for_optimiser(joint_probability, marginal_size, method.optimiser)
end

function _max_ent(counts::Array{<:Integer}, marginal_size::S, method::Direct; precalculated_entropies = Dict())::EMResult where S<:Integer
	return _max_ent(counts ./ sum(counts), marginal_size, method)
end

function _max_ent(joint_probability::Array{T}, marginal_size::S, method::PolymatroidEntropyMethod;precalculated_entropies = Dict{Vector{Integer}, Real}())::EMFMEResult where {T<:Real} where S<:Integer
	res = polymatroid_optim(method, joint_probability, marginal_size; precalculated_entropies)
	ents = Dict{Vector{Int64}, Real}()
	for k in res[4]
		ents[k[1]]=res[2][k[2]]
    end
	return EMFMEResult(res[1], ents)
end

"""
	connected_information(joint_probability::Array{T}, orders::Vector{<:Integer}, method::AbstractEntropyMethod; precalculated_entropies = Dict{Vector{Integer}, Real}(), full_output::Bool=false) where T <: Real -> Tuple{Dict{Int, Float64}, Dict{Int, EResult}}

When called with an EntropyMethod fixes the marginal distribution entropies for all marginals up to order `marginal_size` and maximises the probability of the distribution. When used with polymatroid-based methods reuses cached entropies across orders.

# Arguments
- `method::AbstractEntropyMethod = RawPolymatroid()`:
	- `Direct([optimiser])`: Non-Linear Programming optimisation. The optimiser can be either "ipopt" or "madnlp".
	- `RawPolymatroid([mle_correction, zhang_yeung, optimiser])`: Polymatroid optimisation with naive estimate of marginal entropies. Optionally include MLE correction and Zhang–Yeung inequalities. The optimiser can be an instance of "SCS" or "Mosek".
	- `GPolymatroid([zhang_yeung, optimiser, tolerance])`: Polymatroid optimisation with Grassberger entropy estimator. Optionally include Zhang–Yeung inequalities. The optimiser can be an instance of "SCS" or "Mosek". The tolerance is a relaxation of the constraints to help convergence, can lead to negative CI.

# Keywords
- `precalculated_entropies::Dict = Dict{Vector{Int}, Real}()`: Optional cache to speed up repeated entropy evaluations. Entropies should be computed using log2. See also: `precompute_entropies()`.

# Throws
- `DomainError` if any `orders[i] > ndims(unnormalized)` or if any `orders[i] < 2`.

If a required entropy is `NaN` for some order, a warning is printed and `NaN` is returned for that order.
"""
function connected_information(joint_probability::Array{T}, orders::Vector{<:Integer}, method::AbstractEntropyMethod; precalculated_entropies = Dict{Vector{Integer}, Real}(), full_output::Bool=false)::Tuple{Dict{Int, Float64}, Dict{Int, EResult}} where T <: Real
	if joint_probability isa Array{<:AbstractFloat}
		if ! ((method isa RawPolymatroid) || (method isa Direct))
			throw(DomainError("Method must be either RawPolymatroid or Direct for fixed entropy maximisation and normalized joint distribution."))
		end # better failing soon with a better error message
		!isapprox(sum(joint_probability), 1) &&
			throw(DomainError("Joint probability has to sum to 1"))
	end
	sort!(orders)

	max_size = orders[end]
	min_size = orders[1]

	max_size > ndims(joint_probability) &&
		throw(DomainError("Marginal size cannot be greater than number of dimensions of joint probability"))
	min_size < 2 &&
		throw(DomainError("Marginal size for connected information cannot be less than 2"))

	set_marginals = Set([orders..., (orders .- 1)...])

	dict_results = _max_entropy_for_set(joint_probability, set_marginals, method; precalculated_entropies, full_output)


	ret_dict = Dict{Int, Float64}()

	for m in orders
		if (isnan(dict_results[m].entropy) || isnan(dict_results[m-1].entropy))
			println("WARNING, order $m or $m-1 has NaN entropy, skipping...")
			ret_dict[m] = NaN
			continue
		end
		ret_dict[m] = dict_results[m-1].entropy - dict_results[m].entropy
	end

	return ret_dict, dict_results
end

"""
	connected_information(counts::Array{<:Integer}, orders::Vector{<:Integer}; precalculated_entropies = Dict{Vector{Integer}, Real}()) -> Tuple{Dict{Int, Float64}, Dict{Int, Float64}}

When called with an array of integers assume fixed marginal distributions entropies and defaults to the RawPolymatroid method.
"""
function connected_information(counts::Array{<:Integer}, orders::Vector{<:Integer}; precalculated_entropies = Dict{Vector{Integer}, Real}(), full_output::Bool=false)::Tuple{Dict{Int, Float64}, Dict{Int, EMFMEResult}}
	return connected_information(counts, orders, RawPolymatroid(); precalculated_entropies = precalculated_entropies, full_output = full_output)
end

function connected_information(joint_probability::Array{T}, orders::S, method::AbstractMaximizationMethod; precalculated_entropies = Dict{Vector{Integer}, Real}(), full_output::Bool=false)::Tuple{Dict{Int, Float64}, Dict{Int, EResult}} where T <: Real where S <: Integer
	return connected_information(joint_probability, [orders], method; precalculated_entropies = precalculated_entropies, full_output = full_output)
end

function connected_information(joint_probability::Array{T}, orders::S; precalculated_entropies = Dict{Vector{Int}, Real}(), full_output::Bool=false)::Tuple{Dict{Int, Float64}, Dict{Int, EResult}} where T <: Real where S <: Integer
	return connected_information(joint_probability, [orders]; precalculated_entropies = precalculated_entropies, full_output = full_output)
end

function _max_entropy_for_set(joint_probability::Array{<:T}, marginal_size::Set{<:Integer}, method::PolymatroidEntropyMethod; precalculated_entropies = Dict{Vector{Int}, Real}(), full_output::Bool=false)::Dict{Int, EMFMEResult}  where T <: Real
	ent = precalculated_entropies
	si = Dict()
	result = Dict{Int, EMFMEResult}()
	for m in marginal_size
		val, h, ent, si = polymatroid_optim(
			method,
			joint_probability,
			m;
			precalculated_entropies = ent,
			set_to_index = si,
		)
		if full_output
			ents = Dict{Vector{Int64}, Real}()
			for k in si
				ents[k[1]]=h[k[2]]
			end
			result[m] = EMFMEResult(val, ents)
		else
			result[m] = EMFMEResult(val)
		end
	end
	return result
end

function _max_entropy_for_set(joint_probability::Array{T}, marginal_size::Set{<:Integer}, method::Direct; precalculated_entropies = Dict(), full_output::Bool=false)::Dict{Int, EMResult} where T <: Real
	result = Dict{Int, EMResult}()
	for m in marginal_size
		emresult = _max_ent(joint_probability, m, method; precalculated_entropies)
		if full_output
			result[m] = emresult
		else
			result[m] = EMResult(emresult.entropy)
		end
	end
	return result
end

end # module


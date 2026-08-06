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
	maximise_entropy(joint_probability::Array{<:AbstractFloat}, marginal_size::S, method::AbstractMarginalMethod) -> EMResult where S<:Integer

Find the maximum-entropy distribution whose marginals of size `marginal_size` match those of `joint_probability`.

`joint_probability` is an N-dimensional probability table (array) whose elements sum to ≈ 1. The function fixes **all** marginals of order `marginal_size` and maximizes Shannon entropy over the feasible set.

# Arguments
- `joint_probability::Array{<:AbstractFloat}`: N-dimensional probability array. Must be nonnegative and sum to ~1.
- `marginal_size::Int`: Order of the marginals to hold fixed. For example, `2` fixes every pairwise marginal.

# Keywords
- `method::AbstractMarginalMethod = Cone()`:
	- `Cone([optimizer])`: entropy maximization via exponential cone programming.
	- `Gradient(; iterations, optimiser)`: projected-gradient approach.
	- `Ipfp(; iterations)`: iterative proportional fitting (IPFP).

# Returns
- `EMResult`: A result object holding the **max-entropy distribution** and its **entropy**.

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
function maximise_entropy(joint_probability::Array{<:AbstractFloat}, marginal_size::S)::EMResult where S<:Integer
	return maximise_entropy(joint_probability, marginal_size, Cone())
end

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
	connected_information(joint_probability::Array{<:Real}, orders::Vector{Int}, method = Ipfp()) -> Dict{Int,Float64}

Compute connected information for **multiple orders** efficiently.

This method computes the set of entropies needed for all `orders` in a single pass by evaluating both `m` and `m-1` for each requested order.

# Arguments
- `joint_probability::Array{<:Real}`: N-dimensional probability table summing to ~1.
- `orders::Vector{Int}`: Interaction orders to evaluate. Values must satisfy `2 ≤ orders[i] ≤ ndims(joint_probability)`.

# Keywords
- `method = Ipfp()`: Optimisation strategy used inside repeated `maximise_entropy` calls.

# Returns
- `Dict{Int,Float64}`: Mapping `m => I_m` with `I_m = H^(m-1) - H^m`.

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
function connected_information(joint_probability::Array{T}, orders::Vector{<:Integer}, method::AbstractMarginalMethod; precalculated_entropies = Dict{Vector{Int}, Real}())::Tuple{Dict{Int, Float64}, Dict{Int, Float64}} where T <: Real

	sort!(orders)

	max_size = orders[end]
	min_size = orders[1]

	max_size > ndims(joint_probability) &&
		throw(DomainError("Marginal size cannot be greater than number of dimensions of joint probability"))
	min_size < 2 &&
		throw(DomainError("Marginal size for connected information cannot be less than 2"))

	set_marginals = Set([orders..., (orders .- 1)...])
	dict_entropies = Dict{Int, Float64}()

	for m in set_marginals
		entropy = maximise_entropy(joint_probability, m, method).entropy
		dict_entropies[m] = entropy
	end

	ret_dict = Dict{Int, Float64}()

	for m in orders
		entropy1 = dict_entropies[m-1]
		entropy2 = dict_entropies[m]
		ret_dict[m] = entropy1 - entropy2
	end

	return ret_dict, dict_entropies
end

function connected_information(joint_probability::Array{T}, orders::Vector{<:Integer}; precalculated_entropies = Dict{Vector{Int}, Real}())::Tuple{Dict{Int, Float64}, Dict{Int, Float64}} where T <: AbstractFloat
	return connected_information(joint_probability, orders, Ipfp())
end

function connected_information(counts::Array{<:Integer}, orders::Vector{<:Integer}, method::AbstractMarginalMethod; precalculated_entropies = Dict{Vector{Int}, Real}())::Tuple{Dict{Int, Float64}, Dict{Int, Float64}}
	connected_information(counts ./ sum(counts), orders, method)
end

"""
	maximise_entropy(joint_probability::Array{<:Real}, marginal_size::Int, method::AbstractEntropyMethod) -> Real

Return the **maximum entropy** of any probability distribution whose marginals of order `marginal_size` have the same entropy as those of `joint_probability`.

# Arguments
- `joint_probability::Array{<:Real}`: N-dimensional probability table. Must be nonnegative and sum to ~1.
- `marginal_size::Int`: Order of marginals whose **entropy** is kept fixed.
- `method::AbstractEntropyMethod`: Entropy-optimisation strategy (e.g. `Direct`, `RawPolymatroid`).

# Returns
- `Real`: The maximal entropy value.

# Throws
- `DomainError` if `marginal_size > ndims(joint_probability)` or `marginal_size < 1`.
- `DomainError` if `sum(joint_probability)` is not approximately `1`.
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
	connected_information(
		unnormalized::Array{Int},
		orders::Vector{Int},
		method::PolymatroidEntropyMethod;
		precalculated_entropies = Dict{Vector{Int}, Real}(),
	) -> Tuple{Dict{Int,Float64}, Dict{Int,Float64}}

Compute connected information for multiple orders using **count data** (unnormalized). This variant is tailored for polymatroid-based methods and can reuse cached entropies across orders.

# Arguments
- `unnormalized::Array{Int}`: N-dimensional array of counts.
- `orders::Vector{Int}`: Interaction orders to evaluate. Values must satisfy `2 ≤ orders[i] ≤ ndims(unnormalized)`.

# Keywords
- `method::PolymatroidEntropyMethod`: A polymatroid-based optimisation method (`RawPolymatroid` or `GPolymatroid`).
- `precalculated_entropies::Dict = Dict{Vector{Int}, Real}()`: Optional cache to speed up repeated entropy evaluations. Entropies should be computed using log2.

# Returns
- `(I, H)::Tuple{Dict{Int,Float64}, Dict{Int,Float64}}` where
- `I[m] = H^*(m-1) - H^*(m)` is the connected information of order `m`.
- `H[m]` stores the maximum entropy value `H^*(m)` used to compute `I[m]`.

# Throws
- `DomainError` if any `orders[i] > ndims(unnormalized)` or if any `orders[i] < 2`.

If a required entropy is `NaN` for some order, a warning is printed and that order is skipped in the result.
"""
function connected_information(joint_probability::Array{T}, orders::Vector{<:Integer}, method::AbstractEntropyMethod; precalculated_entropies = Dict{Vector{Integer}, Real}())::Tuple{Dict{Int, Float64}, Dict{Int, Float64}} where T <: Real
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

	dict_entropies = _max_entropy_for_set(joint_probability, set_marginals, method; precalculated_entropies)


	ret_dict = Dict{Int, Float64}()

	for m in orders
		if (isnan(dict_entropies[m]) || isnan(dict_entropies[m-1]))
			println("WARNING, order $m or $m-1 has NaN entropy, skipping...")
			ret_dict[m] = NaN
			continue
		end
		entropy1 = dict_entropies[m-1]
		entropy2 = dict_entropies[m]
		ret_dict[m] = entropy1 - entropy2
	end

	return ret_dict, dict_entropies
end


function connected_information(counts::Array{<:Integer}, orders::Vector{<:Integer}; precalculated_entropies = Dict{Vector{Integer}, Real}())::Tuple{Dict{Int, Float64}, Dict{Int, Float64}}
	return connected_information(counts, orders, RawPolymatroid(); precalculated_entropies = precalculated_entropies)
end

function connected_information(joint_probability::Array{T}, orders::S, method::AbstractMaximizationMethod; precalculated_entropies = Dict{Vector{Integer}, Real}())::Tuple{Dict{Int, Float64}, Dict{Int, Float64}} where T <: Real where S <: Integer
	return connected_information(joint_probability, [orders], method; precalculated_entropies = precalculated_entropies)
end

function connected_information(joint_probability::Array{T}, orders::S; precalculated_entropies = Dict{Vector{Int}, Real}())::Tuple{Dict{Int, Float64}, Dict{Int, Float64}} where T <: Real where S <: Integer
	return connected_information(joint_probability, [orders]; precalculated_entropies = precalculated_entropies)
end

function _max_entropy_for_set(joint_probability::Array{<:T}, marginal_size::Set{<:Integer}, method::PolymatroidEntropyMethod; precalculated_entropies = Dict{Vector{Int}, Real}()) where T <: Real
	ent = precalculated_entropies
	si = Dict()
	result = Dict{Int, Float64}()
	for m in marginal_size
		val, h, ent, si = polymatroid_optim(
			method,
			joint_probability,
			m;
			precalculated_entropies = ent,
			set_to_index = si,
		)
		result[m] = val
	end
	return result
end

function _max_entropy_for_set(joint_probability::Array{T}, marginal_size::Set{<:Integer}, method::Direct; precalculated_entropies = Dict())::Dict{Int, Float64} where T <: Real
	result = Dict{Int, Float64}()
	for m in marginal_size
		emresult = _max_ent(joint_probability, m, method; precalculated_entropies)
		result[m] = emresult.entropy
	end
	return result
end

end # module


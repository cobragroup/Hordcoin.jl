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

using Polyhedra
using CDDLib

export Cone
export Gradient
export Ipfp

export Direct
export PolymatroidEntropyMethod
export RawPolymatroid
export GPolymatroid

export maximise_entropy
export maximise_entropy_ent
export connected_information

export distribution_entropy
export permutations_of_length

include("types.jl")
include("utils.jl")
include("Gcorr.jl")
include("ipfp.jl")
include("exponentialCone.jl")
include("projectedGradient.jl")
include("polymatroid.jl")

include("polyhedra.jl")

"""
	maximise_entropy(joined_probability::Array{<:Real}, marginal_size; method::AbstractMarginalMethod = Cone()) -> EMResult

Find the maximum-entropy distribution whose marginals of size `marginal_size` match those of `joined_probability`.

`joined_probability` is an N-dimensional probability table (array) whose elements sum to ≈ 1. The function fixes **all** marginals of order `marginal_size` and maximizes Shannon entropy over the feasible set.

# Arguments
- `joined_probability::Array{<:Real}`: N-dimensional probability array. Must be nonnegative and sum to ~1.
- `marginal_size::Int`: Order of the marginals to hold fixed. For example, `2` fixes every pairwise marginal.

# Keywords
- `method::AbstractMarginalMethod = Cone()`:
	- `Cone([optimizer])`: entropy maximization via exponential cone programming.
	- `Gradient(; iterations, optimiser)`: projected-gradient approach.
	- `Ipfp(; iterations)`: iterative proportional fitting (IPFP).

# Returns
- `EMResult`: A result object holding the **max-entropy distribution** and its **entropy**.

# Throws
- `DomainError` if `marginal_size > ndims(joined_probability)`.
- `DomainError` if `marginal_size < 1`.
- `DomainError` if `sum(joined_probability)` is not approximately `1`.

If `marginal_size == ndims(joined_probability)`, the input is already fully specified; the function returns it unchanged.

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
function maximise_entropy(joined_probability::Array{<:Real}, marginal_size; method::AbstractMarginalMethod = Cone())::EMResult

	marginal_size > ndims(joined_probability) &&
		throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
	marginal_size < 1 &&
		throw(DomainError("Marginal size has to be positive"))
	!isapprox(sum(joined_probability), 1) &&
		throw(DomainError("Joined probability has to sum to 1"))

	if marginal_size == ndims(joined_probability)
		return EMResult(joined_probability)
	end

	marginals = permutations_of_length(marginal_size, ndims(joined_probability))

	return maximise_method(joined_probability, marginals, method)
end

function maximise_method(joined_probability::Array{<:Real}, marginals, method::Cone)
	cone_for_optimiser(joined_probability, marginals, method.optimiser)
end

function maximise_method(joined_probability::Array{<:Real}, marginals, method::Gradient)
	descent(joined_probability, marginals; iterations = method.iterations, optimiser = method.optimiser)
end

function maximise_method(joined_probability::Array{<:Real}, marginals, method::Ipfp)
	ipfp(joined_probability, marginals, iterations = method.iterations)
end


"""
	connected_information(joined_probability::Array{<:Real}, order::Int, method::AbstractMarginalMethod = Ipfp()) -> Float64

Compute **connected information** (a.k.a. multi-information of order `order`) for the given joint distribution.

It is defined as the drop in maximum entropy when moving from fixing all `(order-1)`-wise marginals to fixing all `order`-wise marginals of `joined_probability`:

```
I_order = H^*(order-1) - H^*(order)
```

# Arguments
- `joined_probability::Array{<:Real}`: N-dimensional probability table summing to ~1.
- `order::Int`: Interaction order (must satisfy `2 ≤ order ≤ ndims(joined_probability)`).

# Keywords
- `method::AbstractMarginalMethod = Ipfp()`: Optimisation strategy used inside the two `maximise_entropy` calls.

# Returns
- `Float64`: Connected information of the requested order.

# Throws
- `DomainError` if `order > ndims(joined_probability)` or `order < 2`.

# Example
```julia-repl
julia> x = [0.1 0.4; 0.4 0.1]
2×2 Matrix{Float64}:
	0.1  0.4
	0.4  0.1

julia> connected_information(x, 2; method = Ipfp())
Progress: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:00
0.2780719051126377
```
"""
function connected_information(joined_probability::Array{T}, order::Int, method::AbstractMarginalMethod = Ipfp())::Float64 where T <: Real

	order > ndims(joined_probability) &&
		throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
	order < 2 &&
		throw(DomainError("Marginal size for connected information cannot be less than 2"))

	entropy1 = maximise_entropy(joined_probability, order - 1; method).entropy
	entropy2 = maximise_entropy(joined_probability, order; method).entropy
	return entropy1 - entropy2
end

function connected_information(joined_probability::Array{Int}, order::Int, method::AbstractMarginalMethod)::Float64
	connected_information(joined_probability ./ sum(joined_probability), order, method)
end
"""
	connected_information(joined_probability::Array{<:Real}, orders::Vector{Int}, method = Ipfp()) -> Dict{Int,Float64}

Compute connected information for **multiple orders** efficiently.

This method computes the set of entropies needed for all `orders` in a single pass by evaluating both `m` and `m-1` for each requested order.

# Arguments
- `joined_probability::Array{<:Real}`: N-dimensional probability table summing to ~1.
- `orders::Vector{Int}`: Interaction orders to evaluate. Values must satisfy `2 ≤ orders[i] ≤ ndims(joined_probability)`.

# Keywords
- `method = Ipfp()`: Optimisation strategy used inside repeated `maximise_entropy` calls.

# Returns
- `Dict{Int,Float64}`: Mapping `m => I_m` with `I_m = H^(m-1) - H^m`.

# Throws
- `DomainError` if any `orders[i] > ndims(joined_probability)` or if any `orders[i] < 2`.

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
function connected_information(joined_probability::Array{T}, orders::Vector{Int}, method::AbstractMarginalMethod = Ipfp()) where T <: Real

	sort!(orders)

	max_size = orders[end]
	min_size = orders[1]

	max_size > ndims(joined_probability) &&
		throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
	min_size < 2 &&
		throw(DomainError("Marginal size for connected information cannot be less than 2"))

	set_marginals = Set([orders..., (orders .- 1)...])
	dict_entropies = Dict{Int, Float64}()

	for m in set_marginals
		entropy = maximise_entropy(joined_probability, m; method).entropy
		dict_entropies[m] = entropy
	end

	ret_dict = Dict{Int, Float64}()

	for m in orders
		entropy1 = dict_entropies[m-1]
		entropy2 = dict_entropies[m]
		ret_dict[m] = entropy1 - entropy2
	end

	return ret_dict
end

function connected_information(joined_probability::Array{Int}, orders::Vector{Int}, method::AbstractMarginalMethod)
	connected_information(joined_probability ./ sum(joined_probability), orders, method)
end

export max_ent_fixed_ent_unnormalized
export max_ent_fixed_ent

"""
	max_ent_fixed_ent_unnormalized(
		unnormalized_distribution::Array{<:Int},
		marginal_size::Int,
		method::AbstractEntropyMethod;
		precalculated_entropies = Dict(),
	) -> Real

Return the **maximum entropy** achievable by any distribution (not necessarily normalized) whose marginals of order `marginal_size` have the same entropy as those of `unnormalized_distribution`.

`unnormalized_distribution` is interpreted as a tensor of **counts** (nonnegative integers). Internally, methods may normalize these counts to a probability table when needed.

# Arguments
- `unnormalized_distribution::Array{<:Int}`: N-dimensional array of counts.
- `marginal_size::Int`: Order of marginals whose **entropy** is kept fixed.
- `method::AbstractEntropyMethod`: One of `Direct`, `RawPolymatroid`, or `GPolymatroid`.

# Keywords
- `precalculated_entropies::Dict = Dict()`:
Optional cache used by polymatroid-based methods to accelerate repeated calls.

# Returns
- `Real`: The maximal entropy subject to the constraint described above.

# Throws
- `DomainError` if `marginal_size > ndims(unnormalized_distribution)` or `marginal_size < 1`.
"""
function max_ent_fixed_ent_unnormalized(
	unnormalized_distribution::Array{<:Int},
	marginal_size::Int,
	method::AbstractEntropyMethod;
	precalculated_entropies = Dict{Vector{Int}, Real}(),
)::EResult

	marginal_size > ndims(unnormalized_distribution) &&
		throw(DomainError("Marginal size cannot be greater than number of dimensions of the distribution"))
	marginal_size < 1 &&
		throw(DomainError("Marginal size has to be positive"))

	return _max_ent_unnormalized(unnormalized_distribution, marginal_size, method; precalculated_entropies)
end

function _max_ent_unnormalized(unnormalized_distr, marginal_size::Int, method::RawPolymatroid; precalculated_entropies = Dict())::EMFMEResult
	res = polymatroid_most_gen(
		method,
		unnormalized_distr,
		marginal_size,
	)
	ents = Dict{Vector{Int64}, Real}()
	for k in res[4]
		ents[k[1]]=res[2][k[2]]
    end
	return EMFMEResult(res[1], ents)
end

function _max_ent_unnormalized(unnormalized_distr, marginal_size::Int, method::Direct; precalculated_entropies = Dict())::EMResult
	joined_prob = unnormalized_distr ./ sum(unnormalized_distr)
	return nlp_entropies_for_optimiser(joined_prob, marginal_size, method.optimiser)
end

function _max_ent_unnormalized(unnormalized_distr, marginal_size::Int, method::GPolymatroid; precalculated_entropies = Dict{Vector{Int}, Real}())::EMFMEResult
	res = polymatroid_most_gen(
		method,
		unnormalized_distr,
		marginal_size;
		precalculated_entropies = precalculated_entropies,
	)
	ents = Dict{Vector{Int64}, Real}()
	for k in res[4]
		ents[k[1]]=res[2][k[2]]
    end
	return EMFMEResult(res[1], ents)
end


"""
	max_ent_fixed_ent(joined_probability::Array{<:Real}, marginal_size::Int, method::AbstractEntropyMethod) -> Real

Return the **maximum entropy** of any probability distribution whose marginals of order `marginal_size` have the same entropy as those of `joined_probability`.

# Arguments
- `joined_probability::Array{<:Real}`: N-dimensional probability table. Must be nonnegative and sum to ~1.
- `marginal_size::Int`: Order of marginals whose **entropy** is kept fixed.
- `method::AbstractEntropyMethod`: Entropy-optimisation strategy (e.g. `Direct`, `RawPolymatroid`).

# Returns
- `Real`: The maximal entropy value.

# Throws
- `DomainError` if `marginal_size > ndims(joined_probability)` or `marginal_size < 1`.
- `DomainError` if `sum(joined_probability)` is not approximately `1`.
"""
function max_ent_fixed_ent(joined_probability::Array{<:Real}, marginal_size::Int, method::AbstractEntropyMethod)::EResult
	if ! ((method isa RawPolymatroid) || (method isa Direct))
		throw(DomainError("Method must be either RawPolymatroid or Direct for fixed entropy maximisation and normalized distribution."))
	end

	marginal_size > ndims(joined_probability) &&
		throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
	marginal_size < 1 &&
		throw(DomainError("Marginal size has to be positive"))
	!isapprox(sum(joined_probability), 1) &&
		throw(DomainError("Joined probability has to sum to 1"))

	return _max_ent(joined_probability, marginal_size, method)
end

function _max_ent(joined_probability::Array{Float64}, marginal_size::Int, method::Direct)::EMResult
	return nlp_entropies_for_optimiser(joined_probability, marginal_size, method.optimiser)
end

function _max_ent(joined_probability::Array{Float64}, marginal_size::Int, method::RawPolymatroid)::EMFMEResult
	res = polymatroid_most_gen(method, joined_probability, marginal_size)
	ents = Dict{Vector{Int64}, Real}()
	for k in res[4]
		ents[k[1]]=res[2][k[2]]
    end
	return EMFMEResult(res[1], ents)
end

# export connected_information



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
function connected_information(unnormalized::Array{Int}, orders::Vector{Int}, method::AbstractEntropyMethod = RawPolymatroid(); precalculated_entropies = Dict{Vector{Int}, Real}())::Tuple{Dict{Int, Float64}, Dict{Int, Float64}}

	sort!(orders)

	max_size = orders[end]
	min_size = orders[1]

	max_size > ndims(unnormalized) &&
		throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
	min_size < 2 &&
		throw(DomainError("Marginal size for connected information cannot be less than 2"))

	set_marginals = Set([orders..., (orders .- 1)...])

	dict_entropies = _max_entropy_unnormalized_for_set(unnormalized, set_marginals, method; precalculated_entropies)


	ret_dict = Dict{Int, Float64}()

	for m in orders
		if (isnan(dict_entropies[m]) || isnan(dict_entropies[m-1]))
			println("WARNING, order $m or $m-1 has NaN entropy, skipping...")
			continue
		end
		entropy1 = dict_entropies[m-1]
		entropy2 = dict_entropies[m]
		ret_dict[m] = entropy1 - entropy2
	end

	return ret_dict, dict_entropies
end

function connected_information(unnormalized::Array{Int}, orders::Int, method::AbstractEntropyMethod = RawPolymatroid(); precalculated_entropies = Dict{Vector{Int}, Real}())::Tuple{Dict{Int, Float64}, Dict{Int, Float64}}
	return connected_information(unnormalized, [orders], method; precalculated_entropies = precalculated_entropies)
end

function _max_entropy_unnormalized_for_set(unnormalized_distribution::Array{<:Int}, marginal_size::Set{<:Int}, method::PolymatroidEntropyMethod; precalculated_entropies = Dict{Vector{Int}, Real}())
	ent = precalculated_entropies
	si = Dict()
	result = Dict{Int, Float64}()
	for m in marginal_size
		val, h, ent, si = polymatroid_most_gen(
			method,
			unnormalized_distribution,
			m;
			precalculated_entropies = ent,
			set_to_index = si,
		)
		result[m] = val
	end
	return result
end

function _max_entropy_unnormalized_for_set(unnormalized_distribution::Array{<:Int}, marginal_size::Set{<:Int}, method::Direct; precalculated_entropies = Dict())
	result = Dict{Int, Float64}()
	for m in marginal_size
		emresult = _max_ent_unnormalized(unnormalized_distribution, m, method; precalculated_entropies)
		result[m] = emresult.entropy
	end
	return result
end



"""
	connected_information(
		normalized::Array{Int},
		orders::Vector{Int},
		method::PolymatroidEntropyMethod;
		precalculated_entropies = Dict{Vector{Int}, Real}(),
	) -> Tuple{Dict{Int,Float64}, Dict{Int,Float64}}

Compute connected information for multiple orders using **count data** (normalized). This variant is tailored for polymatroid-based methods and can reuse cached entropies across orders.

# Arguments
- `normalized::Array{Int}`: N-dimensional array of counts.
- `orders::Vector{Int}`: Interaction orders to evaluate. Values must satisfy `2 ≤ orders[i] ≤ ndims(normalized)`.
- `method::PolymatroidEntropyMethod`: A polymatroid-based optimisation method (`RawPolymatroid` or `GPolymatroid`).

# Keywords
- `precalculated_entropies::Dict = Dict{Vector{Int}, Real}()`: Optional cache to speed up repeated entropy evaluations. Entropies should be computed using log2.

# Returns
- `(I, H)::Tuple{Dict{Int,Float64}, Dict{Int,Float64}}` where
- `I[m] = H^*(m-1) - H^*(m)` is the connected information of order `m`.
- `H[m]` stores the maximum entropy value `H^*(m)` used to compute `I[m]`.

# Throws
- `DomainError` if any `orders[i] > ndims(normalized)` or if any `orders[i] < 2`.

If a required entropy is `NaN` for some order, a warning is printed and that order is skipped in the result.
"""
function connected_information(normalized::Array{T}, orders::Vector{Int}, method::AbstractEntropyMethod; precalculated_entropies = Dict{Vector{Int}, Real}())::Tuple{Dict{Int, Float64}, Dict{Int, Float64}} where T <: AbstractFloat
	if ! ((method isa RawPolymatroid) || (method isa Direct))
		throw(DomainError("Method must be either RawPolymatroid or Direct for fixed entropy connected information and normalized distribution."))
	end
	sort!(orders)

	max_size = orders[end]
	min_size = orders[1]

	max_size > ndims(normalized) &&
		throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
	min_size < 2 &&
		throw(DomainError("Marginal size for connected information cannot be less than 2"))

	set_marginals = Set([orders..., (orders .- 1)...])

	dict_entropies = _max_entropy_normalized_for_set(normalized, set_marginals, method; precalculated_entropies)


	ret_dict = Dict{Int, Float64}()

	for m in orders
		if (isnan(dict_entropies[m]) || isnan(dict_entropies[m-1]))
			println("WARNING, order $m or $m-1 has NaN entropy, skipping...")
			continue
		end
		entropy1 = dict_entropies[m-1]
		entropy2 = dict_entropies[m]
		ret_dict[m] = entropy1 - entropy2
	end

	return ret_dict, dict_entropies
end

function connected_information(normalized::Array{T}, orders::Int, method::AbstractEntropyMethod; precalculated_entropies = Dict{Vector{Int}, Real}())::Tuple{Dict{Int, Float64}, Dict{Int, Float64}} where T <: AbstractFloat
	return connected_information(normalized, [orders], method; precalculated_entropies = precalculated_entropies)
end

function _max_entropy_normalized_for_set(normalized_distribution::Array{<:T}, marginal_size::Set{<:Int}, method::RawPolymatroid; precalculated_entropies = Dict{Vector{Int}, Real}()) where T <: AbstractFloat
	ent = precalculated_entropies
	si = Dict()
	result = Dict{Int, Float64}()
	for m in marginal_size
		val, h, ent, si = polymatroid_most_gen(
			method,
			normalized_distribution,
			m;
			precalculated_entropies = ent,
			set_to_index = si,
		)
		result[m] = val
	end
	return result
end

function _max_entropy_normalized_for_set(normalized_distribution::Array{<:T}, marginal_size::Set{<:Int}, method::Direct; precalculated_entropies = Dict()) where T <: AbstractFloat
	result = Dict{Int, Float64}()
	for m in marginal_size
		emresult = _max_ent(normalized_distribution, m, method)
		result[m] = emresult.entropy
	end
	return result
end

end # module


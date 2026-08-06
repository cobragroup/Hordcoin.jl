# polymatroid.jl


"""
	polymatroid_optim(
		method::PolymatroidEntropyMethod,
		data::Array{Int},
		marginal_size::Int;
		precalculated_entropies::Dict{Vector{Int},<:Real} = Dict(),
		set_to_index::Dict{Vector{Int},Int}} = Dict(),
	) -> Tuple{Real, Vector{Float64}, Dict{Vector{Int},Real}, Dict{Vector{Int},Int}}

General polymatroid optimizer that works directly with **count data** and supports caching.

Builds a JuMP model with polymatroid constraints and sets `h(S)` equal to **entropy estimates** of the corresponding marginals for all `|S| ≤ marginal_size`. The entropy estimator depends on `method`:
- `RawPolymatroid(joint_probabilityability, mle_correction, ...)`: uses `distribution_entropy` on **normalized** `joint_probabilityability` supplied in the method, plus optional `mle_correction`.
- `GPolymatroid(...)`: uses the **Grassberger** estimator `Gcorr(·)` on the count marginals from `data`.

# Arguments
- `method::PolymatroidEntropyMethod`: Estimation strategy and optimizer wrapper (e.g., `RawPolymatroid`, `GPolymatroid`).
- `data::Array{Int}`: N-dimensional **counts** tensor (nonnegative integers).
- `marginal_size::Int`: Largest subset size whose entropy is constrained.

# Keywords
- `precalculated_entropies`: Optional cache mapping subset `Vector{Int}` → entropy value; will be **read and updated**.
- `set_to_index`: Optional mapping from subset to index in `h`; allows reusing the same indexing across calls.

# Returns
- `(Hmax, h_vals, entropies, set_index)` where:
  - `Hmax::Real` is the optimal value of `h(N)`.
  - `h_vals::Vector{Float64}` are optimized values of `h(·)`.
  - `entropies::Dict{Vector{Int},Real}` contains the (possibly cached) entropy values used for each constrained subset.
  - `set_index::Dict{Vector{Int},Int}` is the subset-to-index map used for `h_vals`.

# Throws
- May throw if the optimization model is infeasible or the estimator fails for the provided data.

# Notes
- When `method isa GPolymatroid` and `method.tolerance > 0`, constraints are relaxed to an interval `(1±tolerance)·entropy(S)` instead of equality.
- If `method.zhang_yeung` is `true` and `ndims(data) ≥ 4`, Zhang–Yeung inequalities are added.
"""
function polymatroid_optim(method::PolymatroidEntropyMethod,
	joint_probability::Array{T},
	marginal_size::S;
	precalculated_entropies = Dict{Vector{<:Integer}, Real}(),
	set_to_index = Dict()) where T <: Real where S <: Integer

	model = Model(typeof(method.optimiser))
	set_silent(model)

	num_dimensions = ndims(joint_probability)

	N = 1:num_dimensions

	# dictionary set to index
	s_i = set_to_index
	ent = precalculated_entropies

	index = maximum(values(s_i), init = 0) + 1

	# initialization of  non-negativity constraints
	# 𝒉(𝐴) ≥ 0, ∀𝐴 ∈ 𝒫(𝑁)
	@variable(model, h[1:(2^num_dimensions)] >= 0)

	for A in powerset(N)
		if !haskey(s_i, A)
			s_i[A] = index
			index += 1
		end
	end

	# ∀𝐴 ⊆ 𝒫(𝑁)
	for A in powerset(N)
		if length(A) > num_dimensions - 2
			continue
		end

		# submodularity 
		# 𝒉(𝐴 ∪ 𝘪) + 𝒉(𝐴 ∪ 𝘫) ≥ 𝒉(𝐴 ∪ 𝘪𝘫) + 𝒉(𝐴)
		for ij in powerset(setdiff(N, A), 2, 2)
			i, j = ij
			@constraint(model, h[s_i[sort(A ∪ i)]] + h[s_i[sort(A ∪ j)]] >= h[s_i[sort(A ∪ ij)]] + h[s_i[A]])
		end
	end

	# monotonicity
	# 𝒉(𝑁) ≥ 𝒉(𝑁 ∖ 𝘪), ∀𝑖 ∈ 𝑁
	for i in N
		@constraint(model, h[s_i[N]] >= h[s_i[setdiff(N, i)]])
	end

	~(s::Tuple) = (i for i ∈ 1:num_dimensions if i ∉ s)

	ent_con = Array{Any, 1}(undef, marginal_size)

	for i in 1:marginal_size
		marginals = permutations_of_length(i, num_dimensions)
		for m in marginals
			if !haskey(ent, collect(m))
				ent[collect(m)] = entropy(joint_probability, method, ~(m))
			end
		end
		ent_con[i] = Array{Any, 1}(undef, length(marginals))
		for j in 1:length(marginals)
			m = marginals[j]
			#ent_con[i][j] = @constraint(model, h[s_i[collect(m)]] == ent[s_i[collect(m)]])
			if (method isa GPolymatroid && method.tolerance > 0)
				@constraint(model, h[s_i[collect(m)]] >= (1 - method.tolerance) * ent[collect(m)])
				@constraint(model, h[s_i[collect(m)]] <= (1 + method.tolerance) * ent[collect(m)])
			else
				@constraint(model, h[s_i[collect(m)]] == ent[collect(m)])
			end
		end
	end


	# Zhang-Yeung
	if num_dimensions >= 4 && method.zhang_yeung
		for i in N
			for j in N
				if i == j
					continue
				end
				for kl in powerset(setdiff(N, [i, j]), 2, 2)
					k, l = kl

					ij = s_i[sort(i ∪ j)]
					ik = s_i[sort(i ∪ k)]
					il = s_i[sort(i ∪ l)]
					jk = s_i[sort(j ∪ k)]
					jl = s_i[sort(j ∪ l)]
					kl = s_i[sort(k ∪ l)]
					ikl = s_i[sort(i ∪ k ∪ l)]
					jkl = s_i[sort(j ∪ k ∪ l)]

					i_i = s_i[[i]]
					i_k = s_i[[k]]
					i_l = s_i[[l]]

					@constraint(
						model,
						3 * (h[ik] + h[il] + h[kl])
						+ h[jk] + h[jl] - h[i_i]
						-
						2 * (h[i_k] + h[i_l]) - h[ij]
						-
						4 * h[ikl] - h[jkl] ≥ 0)
				end
			end
		end
	end

	# 𝘩(∅) = 0
	@constraint(model, h[s_i[[]]] == 0)

	@objective(model, Max, h[s_i[N]])

	optimize!(model)

	# TODO: JuMP bug - method not found, but should exist
	#if (!is_solved_and_feasible(model))
	#    throw(DomainError("Model is not feasible with method $(method) and marginal size $(marginal_size)"))
	#end

	return objective_value(model), value.(h), ent, s_i

end

"""
	entropy(joint_probability::Array{Int}, method::RawPolymatroid, inverse_marginals) -> Real

Entropy helper for `RawPolymatroid`.

Computes `distribution_entropy(sum(method.joint_probabilityability, dims = inverse_marginals)) + method.mle_correction` where `inverse_marginals` selects the axes **to sum out**.

# Arguments
- `joint_probability::Array{Int}`: Ignored by this estimator; present for a uniform signature.
- `method::RawPolymatroid`: Contains `mle_correction` field.
- `inverse_marginals`: Iterable of axes to marginalize out.

# Returns
- `Real`: Estimated entropy of the marginal defined by `inverse_marginals`.
"""
function entropy(counts::Array{<:Integer}, method::RawPolymatroid, inverse_marginals)::Real
	p = counts ./ sum(counts)
	if method.mle_correction
		mle_correction = (length(counts) - 1) / (2 * sum(counts))
	else
		mle_correction = 0
	end
	return distribution_entropy(sum(p, dims = inverse_marginals)) + mle_correction
end

function entropy(joint_probability::Array{T}, method::RawPolymatroid, inverse_marginals)::Real where T <: AbstractFloat
	return distribution_entropy(sum(joint_probability, dims = inverse_marginals))
end

"""
	entropy(counts::Array{Int}, method::GPolymatroid, inverse_marginals) -> Real

Entropy helper for `GPolymatroid` using the **Grassberger estimator** on counts.

# Arguments
- `counts::Array{Int}`: Counts tensor.
- `method::GPolymatroid`: Grassberger-based estimator configuration.
- `inverse_marginals`: Iterable of axes to marginalize out.

# Returns
- `Real`: Grassberger estimate `Gcorr(sum(counts, dims = inverse_marginals))`.
"""
function entropy(counts::Array{Int}, method::GPolymatroid, inverse_marginals)::Real
	return Gcorr(sum(counts, dims = inverse_marginals))
end

export precompute_entropies

"""
	precompute_entropies(joint_probability::Array{Int}) -> Dict{Vector{Int},Real}

Pre-compute Grassberger **marginal entropies** for all non-empty subsets of variables in `joint_probability`.

Useful when repeatedly solving polymatroid programs with the same joint_probabilityset: pass the returned dictionary to `precalculated_entropies` in `polymatroid_optim`.

# Arguments
- `joint_probability::Array{Int}`: N-dimensional counts tensor.

# Returns
- `Dict{Vector{Int},Real}` mapping each subset `S` (stored as a sorted `Vector{Int}`) to `Gcorr` entropy of the marginal over `S`.
"""
function precompute_entropies(joint_probability::Array{<:Real}, method::PolymatroidEntropyMethod = GPolymatroid())::Dict{Vector{Int64},Real}

	entropies = Dict()

	num_dimensions = ndims(joint_probability)

	~(s::Tuple) = (i for i ∈ 1:ndims(joint_probability) if i ∉ s)

	for i in 1:num_dimensions
		marginals = permutations_of_length(i, num_dimensions)
		for m in marginals
			entropies[collect(m)] = entropy(joint_probability, method, ~(m))
		end
	end

	return entropies

end

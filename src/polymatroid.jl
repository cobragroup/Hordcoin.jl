# polymatroid.jl

"""
	polymatroid_optim(
		joined_prob::Array{Float64},
		marginal_size;
		model::Model = Model(Mosek.Optimizer),
		zhang_yeung::Bool = false,
		mle_correction::Real = 0,
	) -> Tuple{Real, Vector{Float64}, Dict}

Solve a **polymatroid-based** upper bound on maximum entropy subject to fixed marginal entropies up to order `marginal_size`.

This builds a JuMP model with the polymatroid (Shannon-type) constraints on the set function `h(·)`:
- Nonnegativity: `h(A) ≥ 0` for all subsets `A`.
- Submodularity: `h(A∪i) + h(A∪j) ≥ h(A∪{i,j}) + h(A)`.
- Monotonicity: `h(N) ≥ h(N \\ i)` for all variables `i`.
It then **matches** `h(S)` to the empirical/true marginal entropies of `joined_prob` for all `|S| ≤ marginal_size` and maximizes `h(N)`.

# Arguments
- `joined_prob::Array{Float64}`: N-dimensional **probability table** (nonnegative, sum ≈ 1).
- `marginal_size::Int`: Largest subset size whose entropy is constrained to equal the entropy of the corresponding marginal of `joined_prob`.

# Keywords
- `model::Model = Model(Mosek.Optimizer)`: JuMP model/optimizer to use.
- `zhang_yeung::Bool = false`: If `true`, add a family of **Zhang–Yeung** non-Shannon inequalities (requires `ndims(joined_prob) ≥ 4`).
- `mle_correction::Real = 0`: Additive bias correction applied to each marginal entropy constraint (e.g., Miller–Madow).

# Returns
- `(Hmax, h_vals, set_to_index)` where:
  - `Hmax::Real` is the optimal value of `h(N)`.
  - `h_vals::Vector{Float64}` are the optimized values of `h(·)` indexed by `set_to_index`.
  - `set_to_index::Dict{Vector{Int},Int}` maps each subset `A ⊆ N` (stored as a sorted `Vector{Int}`) to its index in `h_vals`.

# Throws
- May throw if the optimization model is infeasible for the supplied constraints/optimizer.
"""
function polymatroid_optim(joined_prob::Array{Float64}, marginal_size; model::Model = Model(Mosek.Optimizer), zhang_yeung = false, mle_correction = 0)

	set_silent(model)

	num_dimensions = ndims(joined_prob)

	N = 1:num_dimensions

	# doctionary set to index
	s_i = Dict()

	index = 1

	# initialization of  non-negativity constraints
	# 𝒉(𝐴) ≥ 0, ∀𝐴 ∈ 𝒫(𝑁)
	@variable(model, h[1:(2^num_dimensions)] >= 0)

	for A in powerset(N)
		s_i[A] = index
		index += 1
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

	~(s::Tuple) = (i for i ∈ 1:ndims(joined_prob) if i ∉ s)

	for i in 1:marginal_size
		marginals = permutations_of_length(i, num_dimensions)
		@constraint(model, [m in marginals], distribution_entropy(sum(joined_prob, dims = ~(m))) + mle_correction == h[s_i[collect(m)]])
	end

	# Zhang-Yeung
	if num_dimensions >= 4 && zhang_yeung
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

	model

	optimize!(model)

	return objective_value(model), value.(h), s_i

end

"""
	polymatroid_most_gen(
		method::PolymatroidEntropyMethod,
		data::Array{Int},
		marginal_size::Int;
		precalculated_entropies::Dict{Vector{Int},<:Real} = Dict(),
		set_to_index::Dict{Vector{Int},Int}} = Dict(),
	) -> Tuple{Real, Vector{Float64}, Dict{Vector{Int},Real}, Dict{Vector{Int},Int}}

General polymatroid optimizer that works directly with **count data** and supports caching.

Builds a JuMP model with polymatroid constraints and sets `h(S)` equal to **entropy estimates** of the corresponding marginals for all `|S| ≤ marginal_size`. The entropy estimator depends on `method`:
- `RawPolymatroid(joined_probability, mle_correction, ...)`: uses `distribution_entropy` on **normalized** `joined_probability` supplied in the method, plus optional `mle_correction`.
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
function polymatroid_most_gen(method::PolymatroidEntropyMethod,
	data::Array{Int},
	marginal_size::Int;
	precalculated_entropies = Dict{Vector{Int}, Real}(),
	set_to_index = Dict())

	model = Model(typeof(method.optimiser))
	set_silent(model)

	num_dimensions = ndims(data)

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

	~(s::Tuple) = (i for i ∈ 1:ndims(data) if i ∉ s)

	ent_con = Array{Any, 1}(undef, marginal_size)

	for i in 1:marginal_size
		marginals = permutations_of_length(i, num_dimensions)
		for m in marginals
			if !haskey(ent, collect(m))
				ent[collect(m)] = entropy(data, method, ~(m))
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
	entropy(data::Array{Int}, method::RawPolymatroid, inverse_marginals) -> Real

Entropy helper for `RawPolymatroid`.

Computes `distribution_entropy(sum(method.joined_probability, dims = inverse_marginals)) + method.mle_correction` where `inverse_marginals` selects the axes **to sum out**.

# Arguments
- `data::Array{Int}`: Ignored by this estimator; present for a uniform signature.
- `method::RawPolymatroid`: Contains `joined_probability` and `mle_correction` fields.
- `inverse_marginals`: Iterable of axes to marginalize out.

# Returns
- `Real`: Estimated entropy of the marginal defined by `inverse_marginals`.
"""
function entropy(data::Array{Int}, method::RawPolymatroid, inverse_marginals)::Real
	return distribution_entropy(sum(method.joined_probability, dims = inverse_marginals)) + method.mle_correction
end

"""
	entropy(data::Array{Int}, method::GPolymatroid, inverse_marginals) -> Real

Entropy helper for `GPolymatroid` using the **Grassberger estimator** on counts.

# Arguments
- `data::Array{Int}`: Counts tensor.
- `method::GPolymatroid`: Grassberger-based estimator configuration.
- `inverse_marginals`: Iterable of axes to marginalize out.

# Returns
- `Real`: Grassberger estimate `Gcorr(sum(data, dims = inverse_marginals))`.
"""
function entropy(data::Array{Int}, method::GPolymatroid, inverse_marginals)::Real
	return Gcorr(sum(data, dims = inverse_marginals))
end

export precompute_entropies

"""
	precompute_entropies(data::Array{Int}) -> Dict{Vector{Int},Real}

Pre-compute Grassberger **marginal entropies** for all non-empty subsets of variables in `data`.

Useful when repeatedly solving polymatroid programs with the same dataset: pass the returned dictionary to `precalculated_entropies` in `polymatroid_most_gen`.

# Arguments
- `data::Array{Int}`: N-dimensional counts tensor.

# Returns
- `Dict{Vector{Int},Real}` mapping each subset `S` (stored as a sorted `Vector{Int}`) to `Gcorr` entropy of the marginal over `S`.
"""
function precompute_entropies(data::Array{Int})

	entropies = Dict()

	num_dimensions = ndims(data)

	@show num_dimensions

	~(s::Tuple) = (i for i ∈ 1:ndims(data) if i ∉ s)

	for i in 1:num_dimensions
		@show i
		marginals = permutations_of_length(i, num_dimensions)
		for m in marginals
			entropies[collect(m)] = entropy(data, GPolymatroid(), ~(m))
		end
	end

	return entropies

end

# exponentialCone.jl:

"""
	cone_for_optimiser(joint_probability::Array{Float64}, marginals, optimiser::SCS.Optimizer) -> EMResult

Run the exponential-cone formulation using SCS to find a maximum-entropy distribution whose specified marginals match those of `joint_probability`.

# Parameters
- `joint_probability`: N‑dimensional joint probability table (nonnegative, sums ≈ 1).
- `marginals`: Collection of index tuples (e.g. `[(1,), (2,), (1,2)]`) indicating which marginals to fix.
- `optimiser`: An `SCS.Optimizer` instance (its type is used to build the JuMP `Model`).
"""
function cone_for_optimiser(joint_probability::Array{<:AbstractFloat}, marginals::Vector{Tuple}, optimiser::SCS.Optimizer)::EMResult
	cone_over_probabilities(joint_probability, marginals; model = Model(typeof(optimiser)))
end

"""
	cone_for_optimiser(joint_probability::Array{Float64}, marginals, optimiser::MosekTools.Optimizer) -> EMResult

Same as `cone_for_optimiser` but using Mosek’s exponential-cone solver.

# Parameters
- `joint_probability`: Joint probability table.
- `marginals`: Tuples of dimensions to match as marginals.
- `optimiser`: A `MosekTools.Optimizer` instance (type is used to build the `Model`).
"""
function cone_for_optimiser(joint_probability::Array{<:AbstractFloat}, marginals::Vector{Tuple}, optimiser::MosekTools.Optimizer)::EMResult
	cone_over_probabilities(joint_probability, marginals; model = Model(typeof(optimiser)))
end


"""
	cone_over_probabilities(joint_probability::Array{Float64}, marginals; model::Model = Model(SCS.Optimizer)) -> EMResult

Build and solve the **exponential‑cone** program that maximizes Shannon entropy subject to the given marginal constraints from `joint_probability`.

# Parameters
- `joint_probability`: N‑dimensional joint probability table.
- `marginals`: Collection of index tuples specifying the marginals to enforce.
- `model`: JuMP model to use (defaults to SCS). Only the optimizer type is required.
"""
function cone_over_probabilities(joint_probability::Array{<:AbstractFloat}, marginals::Vector{Tuple}; model::Model = Model(SCS.Optimizer))::EMResult

	# defines the complement of a set of dimension
	~(s::Tuple) = (i for i ∈ 1:ndims(joint_probability) if i ∉ s)

	n = length(joint_probability)

	set_silent(model)

	# define the result probabilities
	@variable(model, p[1:n] >= 0)
	@constraint(model, sum(p) == 1)

	q = reshape(p, size(joint_probability)...)


	# Sum over the complement of a set of dimension must be equal to the sum of the result probabilities
	@constraint(model, [m in marginals], sum(joint_probability, dims = ~(m)) .== sum(q, dims = ~(m)))


	# Max ent. reformulation
	@variable(model, t[1:n])
	@constraint(model, [i = 1:n], [t[i], q[i], 1] in MOI.ExponentialCone())
	@objective(model, Max, sum(t))

	optimize!(model)

	# this function uses natural logarithm, so it is need to take it in account
	return EMResult(objective_value(model) / log(2), value.(q))
end

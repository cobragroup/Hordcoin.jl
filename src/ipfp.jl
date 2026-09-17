# ipfp.jl:

"""
ipfp(joint_probability::Array{Float64}, marginals; iterations::Integer = 1000, tol::Float64 = 1e-10) -> EMResult

Perform **Iterative Proportional Fitting Procedure** (IPFP) to adjust a joint probability table so that it matches a given set of marginal constraints.

Starting from a uniform base distribution of the same shape as `joint_probability`, IPFP iteratively scales slices of the base distribution to match the marginals implied by `joint_probability` for the subsets of variables in `marginals`.

The implementation of the algorithm is based on the one in the Python package Dit (https://github.com/dit/dit).

# Arguments
- `joint_probability::Array{Float64}`: N-dimensional probability table. Must be nonnegative; the marginal constraints are taken from this array.
- `marginals`: Collection (e.g., `Vector{Vector{Int}}`) where each element lists the dimensions that define a marginal to be matched.

# Keywords
- `iterations::Integer = 1000`: Maximum number of full IPFP passes over the list of marginals.
- `tol::Float64 = 1e-10`: The convergence tolerance.

# Returns
- `EMResult`: Object containing the fitted maximum-entropy distribution consistent with the given marginals.

# Example
```julia-repl
julia> joint_distribution = [0.1 0.4; 0.4 0.1];

julia> marginals = [[1], [2]];  # fix univariate marginals

julia> result = ipfp(joint_distribution, marginals; iterations = 50)
Entropy: 2.0
Distribution:
[0.25 0.25; 0.25 0.25]
"""
function ipfp(joint_probability::Array{T}, marginals; iterations::Integer=1000, tol::Float64=1e-10)::EMResult where {T <: AbstractFloat}
    num_dimensions = ndims(joint_probability) 
    ~(s::Tuple) = (i for i ∈ 1:num_dimensions if i ∉ s)

    inverse_marginals = [Tuple(~(g)) for g in marginals]
    constraints = [sum(joint_probability; dims=im) for im in inverse_marginals]

    new_distribution = fill(1/length(joint_probability) , size(joint_probability))

    for _ in 1:iterations
        for (constraint, im) in zip(constraints, inverse_marginals)
            current = sum(new_distribution; dims=im)
            ratio = ifelse.(current .> 0, constraint ./ current, 0.0)
            new_distribution .= new_distribution .* ratio
        end

        max_err = maximum(
            [maximum(abs.(sum(new_distribution; dims=im) .- constraint)) for (constraint, im) in zip(constraints, inverse_marginals)]
        )

        if max_err < tol
            break
        end
    end

    total = sum(new_distribution)
    if total > 0
        new_distribution .= new_distribution / total
    end

    return EMResult(new_distribution)
end
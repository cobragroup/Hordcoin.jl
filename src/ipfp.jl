# ipfp.jl:

"""
    ipfp(joint_probability::Array{Float64}, marginals; iterations::Integer = 10) -> EMResult

Perform **Iterative Proportional Fitting Procedure** (IPFP) to adjust a joint probability table so that it matches a given set of marginal constraints.

Starting from a uniform base distribution of the same shape as `joint_probability`, IPFP iteratively scales slices of the base distribution to match the marginals implied by `joint_probability` for the subsets of variables in `marginals`.

# Arguments
- `joint_probability::Array{Float64}`: N-dimensional probability table. Must be nonnegative; the marginal constraints are taken from this array.
- `marginals`: Collection (e.g., `Vector{Vector{Int}}`) where each element lists the dimensions that define a marginal to be matched.

# Keywords
- `iterations::Integer = 10`: Number of full IPFP passes over the list of marginals.

# Returns
- `EMResult`: Object containing the fitted maximum-entropy distribution consistent with the given marginals.

# Example
```julia-repl
julia> joined = [0.1 0.4; 0.4 0.1];

julia> marginals = [[1], [2]];  # fix univariate marginals

julia> result = ipfp(joined, marginals; iterations = 50)
Entropy: 2.0
Distribution:
[0.25 0.25; 0.25 0.25]
```
"""
function ipfp(joint_probability::Array{<:AbstractFloat}, marginals; iterations = 10)::EMResult

    base1 = fill(1/length(joint_probability) , size(joint_probability))
    base2 = fill(1/length(joint_probability) , size(joint_probability)) 

    order = true

    @showprogress for it in 1:iterations
        for m in marginals
            if order
                b1 = base1
                b2 = base2
            else
                b1 = base2
                b2 = base1
            end
            for i in eachindex(IndexCartesian(), joint_probability)
                # Index that takes slice of joint_probability that corresponds to coordinates according to m
                idx = [id in m ? i[id] : Colon() for id in 1:ndims(joint_probability)]
                # Improve by precounting sum of joint_probability
                s = sum(joint_probability[idx...])
                if s == 0
                    b2[i] = 0
                else
                    b2[i] = b1[i] * s / sum(b1[idx...])
                end
            end
            order = !order  
        end
    end

    return EMResult(order ? base1 : base2)
end

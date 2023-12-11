
using JuMP, MosekTools
using JuMP.Containers: @container

function cone_over_probabilities(joined_prob::Array{Float64}, marginals)

    # defines the complement of a set of dimension
    ~(s::Tuple) = (i for i = 1:ndims(joined_prob) if i ∉ s)

    n = length(joined_prob)

    model = Model(Mosek.Optimizer)
    # ?
    set_silent(model)

    # define the result probabilities
    @variable(model, p[1:n] >= 0)
    @constraint(model, sum(p) == 1)

    q = reshape(p, size(joined_prob)...)


    # Sum over the complement of a set of dimension must be equal to the sum of the result probabilities
    @constraint(model, [m in marginals], sum(joined_prob, dims = ~(m)) .== sum(q, dims = ~(m)))

    
    # Max ent. reformulation
    @variable(model, t[1:n])
    @constraint(model, [i = 1:n], [t[i], q[i], 1] in MOI.ExponentialCone())
    @objective(model, Max, sum(t))

    optimize!(model)

    # this function uses natural logarithm, so it is need to take it in account
    (objective_value(model) / log(2), value.(q)) #round.(value.(q), digits=5))
end

using JuMP, MosekTools
using JuMP.Containers: @container

function project_to_constraints(data, joined_prob::Array{Float64}, marginals)

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
        
    @variable(model, t) # t is min distance
    @constraint(model, [t, (p - data)...] in SecondOrderCone())
    #@constraint(model, [i=1:n], p[i] >= 0)
    @objective(model, Min, t)

    optimize!(model)

    # this function uses natural logarithm, so it is need to take it in account
    # p is sometimes negative!!!!!!!!
    return objective_value(model), value.(p) #round.(value.(q), digits=5))
end

function partial_der_entropy(x; default = 10)
    # ignoring division by log(2)
    if x <= 0
        # default value is important due to the derivative of entropy not being defined at 0
        return default
    end
    return (- log(x) - 1)
end

function descent(data, marginals; iterations = 1000)
    step = 0.01
    flat = vec(data)
    smallest = 1/length(data) * 0.1
    def = -log(smallest) - 1
    @showprogress for i in 1:iterations
        prev = flat
        flat += step * partial_der_entropy.(flat; default = def)
        #@show flat
        distance, flat = project_to_constraints(flat, data, marginals)
        #step *= 0.95
    end
    return reshape(flat, size(data)) 
end
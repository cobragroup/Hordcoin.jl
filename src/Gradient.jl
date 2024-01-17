using JuMP, MosekTools
using JuMP.Containers: @container

function project_to_constraints(data, joined_prob::Array{Float64}, marginals; solver::AbstractOptimizer)

    # defines the complement of a set of dimension
    ~(s::Tuple) = (i for i = 1:ndims(joined_prob) if i ∉ s)

    n = length(joined_prob)

    if solver isa SCSOptimizer
        model = Model(SCS.Optimizer)
    elseif solver isa MosekOptimizer
        model = Model(Mosek.Optimizer)
    else
        error("Unknown solver of type $(typeof(solver))")
    end
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
    @objective(model, Min, t)

    optimize!(model)

    # this function uses natural logarithm, so it is need to take it in account
    # p is sometimes negative!!!!!!!!
    any(x -> x < 0, value.(q)) && throw(DomainError("Solver wasn't able to solve least square distance without negative probability"))
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

function descent(data, marginals; iterations = 1000, solver::AbstractOptimizer = SCSOptimizer())
    step = 0.01
    flat = vec(data)
    smallest = 1/length(data) * 0.1
    def = -log(smallest) - 1
    @showprogress for i in 1:iterations
        flat += step * partial_der_entropy.(flat; default = def)
        flat[flat .< 0] .= 0
        distance, flat = project_to_constraints(flat, data, marginals; solver = solver)
        #step *= 0.95
    end
    return reshape(flat, size(data)) 
end
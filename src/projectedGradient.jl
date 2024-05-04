# projectedGradient.jl:

function project_to_constraints(data, joined_prob::Array{Float64}, marginals; model::Model = Model(SCS.Optimizer))::Array{Float64}

    # defines the complement of a set of dimension
    ~(s::Tuple) = (i for i = 1:ndims(joined_prob) if i ∉ s)

    n = length(joined_prob)
    
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

    any(x -> x < 0, value.(q)) && 
        throw(DomainError("Solver wasn't able to solve least square distance without negative probability"))
    return value.(p)
end

function project_with_model(data, joined_prob::Array{Float64}, marginals, optimiser::SCS.Optimizer)::Array{Float64}
    project_to_constraints(data, joined_prob, marginals; model = Model(typeof(optimiser)))
end

function project_with_model(data, joined_prob::Array{Float64}, marginals, optimiser::MosekTools.Optimizer)::Array{Float64}
    project_to_constraints(data, joined_prob, marginals; model = Model(typeof(optimiser)))
end



function partial_der_entropy(x::T; default = 10) where {T <: Real}
    if x <= 0
        # default value is important due to the derivative of entropy not being defined at 0
        return convert(T, default)
    end
    return (- log(x) - 1)
end

function descent(data, marginals; iterations = 1000, optimiser::MathOptInterface.AbstractOptimizer)::EMResult
    step = 0.01
    flat = vec(data)

    smallest = 1/length(data) * 0.1
    def = -log(smallest) - 1

    @showprogress for _ in 1:iterations
        flat += step * partial_der_entropy.(flat; default = def)
        flat[flat .< 0] .= 0
        flat = project_with_model(flat, data, marginals, optimiser)
    end

    return EMResult(reshape(flat, size(data)))
end
# projectedGradient.jl:

"""
    project_to_constraints(data, joined_prob::Array{Float64}, marginals; model::Model = Model(SCS.Optimizer)) -> Vector{Float64}

Project a flattened candidate distribution `data` onto the affine set of joint tables whose specified **marginals** match those of `joined_prob`.
Uses a second‑order cone program (least‑squares distance) via JuMP.

- `data`: Flattened probabilities (same length as `joined_prob`).
- `joined_prob`: N‑dimensional joint table providing target marginals.
- `marginals`: Collection of index tuples to be matched (e.g. `[(1,), (2,), (1,2)]`).
- `model`: JuMP `Model` specifying the optimizer.

Returns the projected flat vector of probabilities.
"""
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

"""
    project_with_model(data, joined_prob::Array{Float64}, marginals, optimiser::SCS.Optimizer) -> Vector{Float64}

Shorthand to call `project_to_constraints` using SCS.
"""
function project_with_model(data, joined_prob::Array{Float64}, marginals, optimiser::SCS.Optimizer)::Array{Float64}
    project_to_constraints(data, joined_prob, marginals; model = Model(typeof(optimiser)))
end

"""
    project_with_model(data, joined_prob::Array{Float64}, marginals, optimiser::MosekTools.Optimizer) -> Vector{Float64}

Shorthand to call `project_to_constraints` using Mosek.
"""
function project_with_model(data, joined_prob::Array{Float64}, marginals, optimiser::MosekTools.Optimizer)::Array{Float64}
    project_to_constraints(data, joined_prob, marginals; model = Model(typeof(optimiser)))
end



"""
    partial_der_entropy(x; default=10)

Derivative of `-p log p` at `x` (in nats). For nonpositive `x`, returns the finite fallback `default`.
"""
function partial_der_entropy(x::T; default = 10) where {T <: Real}
    if x <= 0
        # default value is important due to the derivative of entropy not being defined at 0
        return convert(T, default)
    end
    return (- log(x) - 1)
end

"""
    descent(data, marginals; iterations=1000, optimiser) -> EMResult

Projected‑gradient routine for **maximum entropy** with fixed marginals.
Performs gradient ascent on entropy, followed by a projection onto the marginal‑matching set via `project_with_model`.

- `data`: Initial joint table (Array). Also defines the target marginals.
- `marginals`: Collection of index tuples specifying which marginals to enforce.
- `iterations`: Number of gradient–projection steps.
- `optimiser`: Optimizer instance used inside the projection (e.g., `SCS.Optimizer()` or `MosekTools.Optimizer()`).

Returns an `EMResult` with the fitted distribution.
"""
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
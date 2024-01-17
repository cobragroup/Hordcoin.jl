# types.jl:

struct EMResult
    entropy::Float64
    joined_probability::Array{T} where T <: Real
end
EMResult(joined_probability::Array{T}) where T <: Real = 
    EMResult(distribution_entropy(joined_probability), joined_probability)
Base.show(io::IO, result::EMResult) = 
    print(io, "Entropy: ", result.entropy, "\nDistribution:\n", result.joined_probability)


abstract type AbstractOptimizer end

struct SCSOptimizer <: AbstractOptimizer end
struct MosekOptimizer <: AbstractOptimizer end


abstract type AbstractMethod end

struct Cone <: AbstractMethod
    optimizer::AbstractOptimizer
end
Cone() = Cone(SCSOptimizer())

struct Gradient <: AbstractMethod
    iterations::Int
    optimizer::AbstractOptimizer
end
Gradient() = Gradient(10, SCSOptimizer())
Gradient(iterations::Int) = Gradient(iterations, SCSOptimizer())

struct Ipfp <: AbstractMethod
    iterations::Int
end
Ipfp() = Ipfp(10)    

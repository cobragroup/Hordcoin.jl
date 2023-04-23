# types.jl:

struct EMResult
    entropy::Float64
    joined_probability::Array{T} where T <: Real
end
EMResult(joined_probability::Array{T}) where T <: Real = 
    EMResult(distribution_entropy(joined_probability), joined_probability)
Base.show(io::IO, result::EMResult) = 
    print(io, "Entropy: ", result.entropy, "\nDistribution:\n", result.joined_probability)


abstract type AbstractMarginalMethod end

struct Cone <: AbstractMarginalMethod
    optimiser::MathOptInterface.AbstractOptimizer
end
Cone() = Cone(SCS.Optimizer())

struct Gradient <: AbstractMarginalMethod
    iterations::Int
    optimiser::MathOptInterface.AbstractOptimizer
end
Gradient() = Gradient(10, SCS.Optimizer())
Gradient(iterations::Int) = Gradient(iterations, SCS.Optimizer())

struct Ipfp <: AbstractMarginalMethod
    iterations::Int
end
Ipfp() = Ipfp(10)    


abstract type AbstractEntropyMethod end

struct Direct <: AbstractEntropyMethod
    optimiser::String
end

Direct() = Direct("ipopt")

abstract type PolymatroidEntropyMethod <: AbstractEntropyMethod end

mutable struct RawPolymatroid <: PolymatroidEntropyMethod
    mle_correction::Float64
    zhang_yeung::Bool
    optimiser::MathOptInterface.AbstractOptimizer
    joined_probability
end

RawPolymatroid() = RawPolymatroid(0.0, false, SCS.Optimizer(), nothing)
RawPolymatroid(mle_correction::Float64) = RawPolymatroid(mle_correction, false, SCS.Optimizer(), nothing)
RawPolymatroid(zhang_yeung::Bool) = RawPolymatroid(0.0, zhang_yeung, SCS.Optimizer(), nothing)
RawPolymatroid(mle_correction::Float64, zhang_yeung::Bool) = RawPolymatroid(mle_correction, zhang_yeung, SCS.Optimizer(), nothing)
RawPolymatroid(mle_correction::Float64, zhang_yeung::Bool, optimiser::MathOptInterface.AbstractOptimizer) = RawPolymatroid(mle_correction, zhang_yeung, optimiser, nothing)

struct NsbPolymatroid <: PolymatroidEntropyMethod
    zhang_yeung::Bool
    optimiser::MathOptInterface.AbstractOptimizer
    tolerance::Float64
end

NsbPolymatroid() = NsbPolymatroid(false, SCS.Optimizer(), 0)
NsbPolymatroid(zhang_yeung::Bool) = NsbPolymatroid(zhang_yeung, SCS.Optimizer(), 0)
# types.jl
"""
Abstract supertype for structs containing the results of entropy maximisation.
"""
abstract type EResult end

"""
Holds the result of an entropy maximisation, storing the computed entropy value and the corresponding joint probability distribution.

# Parameters
- `entropy::Float64`: The computed entropy value.
- `joint_probability::Array{T}`: The joint probability distribution.
"""
struct EMResult <: EResult
	entropy::Float64
	joint_probability::Array{T} where T <: Real
end

EMResult(joint_probability::Array{T}) where T <: Real =
	EMResult(distribution_entropy(joint_probability), joint_probability)
EMResult(entropy::Float64) =
	EMResult(entropy, Array{Float64}(undef, 0))
Base.show(io::IO, ::MIME"text/plain", result::EMResult) =
	print(io, "Entropy: ", result.entropy, "\nDistribution:\n", result.joint_probability)
Base.show(io::IO, result::EMResult) =
	print(io, "EMResult(", result.entropy, ", ", result.joint_probability, ")")

"""
Holds the result of an entropy maximisation with fixed marginal entropies, storing the computed entropy value and the corresponding marginal entropies.

# Parameters
- `entropy::Float64`: The computed entropy value.
- `marginal_entropies::Dict{Vector{Int64}, T}`: The joint probability distribution.
"""
struct EMFMEResult <: EResult
	entropy::Float64
	marginal_entropies::Dict{Vector{Int64}, T} where T <: Real
end
EMFMEResult(entropy::Float64) =
	EMFMEResult(entropy, Dict{Vector{Int64}, Float64}())
Base.show(io::IO, ::MIME"text/plain", result::EMFMEResult) =
	print(io, "Entropy: ", result.entropy, "\nMarginal Entropies:\n", result.marginal_entropies)
Base.show(io::IO, result::EMFMEResult) =
	print(io, "EMFMEResult(", result.entropy, ", ", result.marginal_entropies, ")")

abstract type AbstractMaximizationMethod end

"""
Abstract supertype for methods used to maximise entropy with fixed marginal constraints.
"""
abstract type AbstractMarginalMethod <: AbstractMaximizationMethod end

"""
Marginal method that uses cone programming via the specified optimiser.

# Parameters
- `optimiser::MathOptInterface.AbstractOptimizer`: Optimiser used for cone programming.
"""
struct Cone <: AbstractMarginalMethod
	optimiser::MathOptInterface.AbstractOptimizer
end
Cone() = Cone(SCS.Optimizer())

"""
Marginal method that uses a gradient-based approach for entropy maximisation.

# Parameters
- `iterations::Int`: Number of iterations to run, defaults to `10`.
- `optimiser::MathOptInterface.AbstractOptimizer`: Optimiser used for gradient updates.
"""
struct Gradient <: AbstractMarginalMethod
	iterations::Int
	optimiser::MathOptInterface.AbstractOptimizer
end
Gradient() = Gradient(10, SCS.Optimizer())
Gradient(iterations::Int) = Gradient(iterations, SCS.Optimizer())

"""
Marginal method that uses the Iterative Proportional Fitting Procedure (IPFP).

# Parameters
- `iterations::Int`: Number of iterations for IPFP, defaults to `10`.
"""
struct Ipfp <: AbstractMarginalMethod
	iterations::Int
end
Ipfp() = Ipfp(10)


"""
Abstract supertype for different entropy maximisation strategies.
"""
abstract type AbstractEntropyMethod <: AbstractMaximizationMethod end

"""
Entropy maximisation method that solves the problem directly via a given optimiser.

# Parameters
- `optimiser::String`: Name of the optimiser to use, defaults to ipopt.
"""
struct Direct <: AbstractEntropyMethod
	optimiser::String
end

Direct() = Direct("ipopt")

"""
Abstract supertype for entropy methods based on polymatroid (submodular) constraints.
"""
abstract type PolymatroidEntropyMethod <: AbstractEntropyMethod end

"""
Polymatroid-based entropy method that uses empirical marginal entropies, with options for MLE correction and Zhang–Yeung inequalities.

# Parameters
- `mle_correction::Float64`: Amount of MLE bias correction to apply (default `0.0`).
- `zhang_yeung::Bool`: Whether to include Zhang–Yeung inequalities (default `false`).
- `optimiser::MathOptInterface.AbstractOptimizer`: Optimiser to use.
"""
mutable struct RawPolymatroid <: PolymatroidEntropyMethod
	mle_correction::Bool
	zhang_yeung::Bool
	optimiser::MathOptInterface.AbstractOptimizer
end

RawPolymatroid() = RawPolymatroid(false, false, SCS.Optimizer())
RawPolymatroid(mle_correction::Bool) = RawPolymatroid(mle_correction, false, SCS.Optimizer())
RawPolymatroid(mle_correction::Bool, zhang_yeung::Bool) = RawPolymatroid(mle_correction, zhang_yeung, SCS.Optimizer())

"""
Polymatroid-based entropy method that uses the Grassberger entropy estimator for marginals.

# Parameters
- `zhang_yeung::Bool`: Whether to include Zhang–Yeung inequalities (default `false`).
- `optimiser::MathOptInterface.AbstractOptimizer`: Optimiser to use.
- `tolerance::Float64`: Relative tolerance for constraints (default `0`).
"""
struct GPolymatroid <: PolymatroidEntropyMethod
	zhang_yeung::Bool
	optimiser::MathOptInterface.AbstractOptimizer
	tolerance::Float64
end

GPolymatroid() = GPolymatroid(false, SCS.Optimizer(), 0)
GPolymatroid(zhang_yeung::Bool) = GPolymatroid(zhang_yeung, SCS.Optimizer(), 0)
GPolymatroid(tolerance::Float64) = GPolymatroid(false, SCS.Optimizer(), tolerance)

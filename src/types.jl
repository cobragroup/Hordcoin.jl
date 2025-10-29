# types.jl

"""
Holds the result of an entropy maximisation, storing the computed entropy value and the corresponding joint probability distribution.

# Parameters
- `entropy::Float64`: The computed entropy value.
- `joined_probability::Array{T}`: The joint probability distribution.
"""
struct EMResult
	entropy::Float64
	joined_probability::Array{T} where T <: Real
end

EMResult(joined_probability::Array{T}) where T <: Real =
	EMResult(distribution_entropy(joined_probability), joined_probability)
Base.show(io::IO, result::EMResult) =
	print(io, "Entropy: ", result.entropy, "\nDistribution:\n", result.joined_probability)


"""
Abstract supertype for methods used to maximise entropy with fixed marginal constraints.
"""
abstract type AbstractMarginalMethod end

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
abstract type AbstractEntropyMethod end

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
- `joined_probability`: Optional probability distribution used for entropy estimation.
"""
mutable struct RawPolymatroid <: PolymatroidEntropyMethod
	mle_correction::Float64
	zhang_yeung::Bool
	optimiser::MathOptInterface.AbstractOptimizer
	joined_probability::Any
end

RawPolymatroid() = RawPolymatroid(0.0, false, SCS.Optimizer(), nothing)
RawPolymatroid(mle_correction::Float64) = RawPolymatroid(mle_correction, false, SCS.Optimizer(), nothing)
RawPolymatroid(zhang_yeung::Bool) = RawPolymatroid(0.0, zhang_yeung, SCS.Optimizer(), nothing)
RawPolymatroid(mle_correction::Float64, zhang_yeung::Bool) = RawPolymatroid(mle_correction, zhang_yeung, SCS.Optimizer(), nothing)
RawPolymatroid(mle_correction::Float64, zhang_yeung::Bool, optimiser::MathOptInterface.AbstractOptimizer) = RawPolymatroid(mle_correction, zhang_yeung, optimiser, nothing)

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

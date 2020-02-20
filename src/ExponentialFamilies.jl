module ExponentialFamilies

using Reexport

@reexport using Distributions
@reexport using StatsBase

using Expectations
using GLM
using LinearAlgebra
using Splines2
using Random


import Base:eltype
import Base.Broadcast: broadcastable
import Base:minimum, maximum, extrema
import Distributions:probs, logpdf, pdf, cf, mgf, insupport, support, sampler
import StatsBase:fit, moment
import Statistics:mean, var
import Random:rand, AbstractRNG


include("discrete_exp_families.jl")
include("continuous_exp_families.jl")


export DiscreteExponentialFamilyModel,
       DiscreteExponentialFamily,
	   ContinuousExponentialFamilyModel,
	   ContinuousExponentialFamily,
	   LindseyMethod

end # module

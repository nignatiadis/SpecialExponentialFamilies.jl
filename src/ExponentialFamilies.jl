module ExponentialFamilies

using Reexport

@reexport using Distributions
@reexport using StatsBase

using LinearAlgebra

using Random
import Random:rand, AbstractRNG

import Statistics:mean, var

import Distributions:probs, pdf, cf, mgf, support, sampler
using StatsBase
import StatsBase:moment

using StatsModels

import Base:eltype
import Base.Broadcast: broadcastable


include("discrete_exp_families.jl")

export DiscreteExponentialFamilyModel,
       DiscreteExponentialFamily


end # module

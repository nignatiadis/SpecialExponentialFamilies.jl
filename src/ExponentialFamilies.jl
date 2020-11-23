module ExponentialFamilies

using Reexport

@reexport using Distributions
@reexport using StatsBase

using Expectations
using Empirikos
using GLM
using LinearAlgebra

using OffsetArrays
using LinearAlgebra
using Statistics

using Random
using Requires

import Base:eltype
import Base.Broadcast: broadcastable
import Base:minimum, maximum, extrema
import Distributions:probs, logpdf, pdf, cf, mgf, insupport, support, sampler
import StatsBase:fit, moment
import Statistics:mean, var
import Random:rand, rand!, AbstractRNG

using UnPack

include("splines.jl")
include("continuous_exp_families.jl")
include("lindsey.jl")

function __init__()
    @require ApproxFun="28f2ccd6-bb30-5033-b560-165f7b14dc2f" include("sample_continuous_exp_families.jl")
end

export ExponentialFamily,
       ExponentialFamilyDistribution,
       LindseyMethod

end # module

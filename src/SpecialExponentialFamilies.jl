module SpecialExponentialFamilies

const SEF = SpecialExponentialFamilies

using Reexport

@reexport using Distributions
using StatsBase
import StatsBase: confint, fit

using DiffResults
using Empirikos
using Expectations

using ForwardDiff
using GLM
using LinearAlgebra

using OffsetArrays
using Optim
using LinearAlgebra
using Statistics

using Random
using Requires

import Base: eltype
import Base.Broadcast: broadcastable
import Base: minimum, maximum, extrema
import Distributions: probs, logpdf, pdf, cf, mgf, insupport, support, sampler
import StatsBase: fit, moment
import Statistics: mean, var
import Random: rand, rand!, AbstractRNG

using UnPack

include("splines.jl")
include("continuous_exp_families.jl")
include("lindsey.jl")
include("logspline_deconvolution.jl")
include("datasets.jl")

function __init__()
    @require ApproxFun = "28f2ccd6-bb30-5033-b560-165f7b14dc2f" include(
        "sample_continuous_exp_families.jl"
    )
end

export ExponentialFamily, ExponentialFamilyDistribution, LindseyMethod, SEF

export fit, confint

end # module

using ApproxFun
using ExponentialFamilies
using Empirikos
using Random
using StatsBase
using Expectations
using Splines2
using GLM
using Plots
using LinearAlgebra
using Expectations
using Test
using QuadGK

unif_measure = Uniform(-3.0,3.0)

_df = 3
expfam = ExponentialFamily(; basemeasure=unif_measure, df=_df)
@test length(expfam.Q(1.0)) == _df


expfam_ones = expfam(ones(_df))
@test quadgk(μ->pdf(expfam_ones, μ), -3, 3)[1] ≈ 1.0
n = 10_000

Random.seed!(10)
Zs = rand(expfam_ones, n)

disc_Z = summarize(discretize(Zs))

@test sum(weights(disc_Z)) == n


lindsey_fit = StatsBase.fit(LindseyMethod(ef=expfam), Zs)
lindsey_fit.α

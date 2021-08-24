using ApproxFun
using SpecialExponentialFamilies
using Empirikos
using Random
using DiffResults
using ForwardDiff
using UnPack
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

Zs_noise = StandardNormalSample.(Zs .+ 1.0 * randn(n))
Zs_noise2 = NormalSample.(response.(Zs_noise), 1.0)
Zs_discr = summarize(discretize(Zs_noise))

tmp = fit(PenalizedMLE(ef=expfam, optim_options=Optim.Options(show_trace=true, show_every=1, iterations=20)), Zs_noise)
tmp2 = fit(PenalizedMLE(ef=expfam, optim_options=Optim.Options(show_trace=true, show_every=1, iterations=20)), Zs_noise2)
tmp3 = fit(PenalizedMLE(ef=expfam, optim_options=Optim.Options(show_trace=true, show_every=1, iterations=20)), Zs_discr)

tmp2.α_opt
tmp3.α_opt
tmp_lindsey = fit(LindseyMethod(ef=expfam), Zs)
tmp.α_opt == tmp2.α_opt

tmp2.α

sqrt.(diag(tmp3.α_covmat))


efm = EFMarginalDistribution3(expfam_ones, StandardNormalSample())
@test quadgk(x -> pdf(efm, x), -Inf, +Inf)[1] ≈ 1.0
@test cdf(efm, Inf) == 1.0
@test ForwardDiff.derivative(x->cdf(efm,x), 1.0) ≈ pdf(efm, 1.0)
@test pdf(expfam_ones, StandardNormalSample(-2.0)) == pdf(efm, -2.0)
#function marginalize(Z::AbstractNormalSample, prior::Normal)
#end

using SpecialExponentialFamilies
using Empirikos
using ApproxFun
using Random
using Intervals

expfamily = ExponentialFamily(basemeasure=Uniform(-4, 6), df=8, scale=true)

n = 10_000
α_smoothed_twin_tower =  [10; 0.0; 6.0; 9.0; -3.0; -2.0; -8.0; -1.0]

smoothed_twin_tower = expfamily(α_smoothed_twin_tower)

Random.seed!(1000)
Zs = rand(smoothed_twin_tower, n)
Zs_normal = StandardNormalSample.(Zs .+ randn(n))

_ef_fit = StatsBase.fit(SEF.PenalizedMLE(ef=expfamily, c0=1e-6), summarize(discretize(Zs_normal)))
_ef_fit.α_opt

ts = -3:0.1:3

lfsrs = Empirikos.PosteriorProbability.(StandardNormalSample.(ts), Intervals.Interval(0,nothing))
postmeans = Empirikos.PosteriorMean.(StandardNormalSample.(ts))

lfsr_cis = confint.(_ef_fit, lfsrs)
postmean_cis = confint.(_ef_fit, postmeans)

#using Plots
#pgfplotsx()
#plot(lfsrs, smoothed_twin_tower)
#plot!(ts, lfsr_cis)

#plot(postmeans, smoothed_twin_tower)
#plot!(ts, postmean_cis)

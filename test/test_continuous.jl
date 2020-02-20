using ExponentialFamilies

using StatsBase
using Expectations
using Splines2
using GLM
using Plots
using LinearAlgebra
using Expectations

unif_measure = Uniform(-3.0,3.0)
theta_grid = collect(-3:0.01:3)

ns_f = ns_(theta_grid; df=6)
mymodel = ContinuousExponentialFamilyModel(unif_measure, ns_f)
mymodel2 = ContinuousExponentialFamilyModel(unif_measure, theta_grid; df=6)

ns_f(1.0)

res = vcat(mymodel2.Q.(theta_grid)'...)
res2 = vcat(mymodel.Q.(theta_grid)'...)

sum(abs2.(res) ;dims=1)

plot(theta_grid, res)
plot(theta_grid, res2)


myexp = expectation(unif_measure; n=20)


plot(diff(myexp.nodes), seriestype=:scatter)

myexp.nodes


αs = zeros(6)


mymeasure = ContinuousExponentialFamily(unif_measure, ns_f, αs)


pdf(mymeasure, 1.0; include_base_measure=false)

logpdf(mymeasure, -Inf; include_base_measure=true)

inf_range = [-Inf; collect(-1:0.1:1); +Inf]

Zs = randn(1000)
hist_inf = fit(Histogram,Zs, inf_range)

StatsBase.midpoints(inf_range)

mymeasure.log_normalizing_constant



Xs = rand(mymeasure.base_measure, 10_000)

myhistogram = fit(Histogram,Xs, nbins=50)


tmp_fit = fit(mymodel, Xs, LindseyMethod(50))
tmp_fit2 = fit(mymodel, myhistogram, LindseyMethod(50))

tmp_fit.α

tmp_fit2.α


# check histogram fitting functionality




tmp_fit.Q(1.0)*

pdf.(tmp_fit, 1.0)
plot(theta_grid, pdf.(tmp_fit, theta_grid), ylim=(0,0.25))

tmp_glm[:hist].weights

tmp_pred = predict(tmp_glm[:fit])

plot(tmp_pred)
coef(tmp_glm[:fit])


extrema(mymodel)






logpdf(mymeasure, theta_grid)

plot(theta_grid, exp.(logpdf(mymeasure, theta_grid)), ylim=(0,0.3))

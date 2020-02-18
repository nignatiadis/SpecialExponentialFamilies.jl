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


myexp =

mymodel = ContinuousExponentialFamilyModel(unif_measure, ns_f)
αs = zeros(6)


mymeasure = ContinuousExponentialFamily(unif_measure, ns_f, αs)

Xs = rand(mymeasure.base_measure, 10_000)

tmp_fit = fit(mymodel, Xs; length=500)


pdf(tmp_fit, [1.0])
plot(theta_grid, pdf(tmp_fit, theta_grid), ylim=(0,0.25))

tmp_glm[:hist].weights

tmp_pred = predict(tmp_glm[:fit])

plot(tmp_pred)
coef(tmp_glm[:fit])


extrema(mymodel)






logpdf(mymeasure, theta_grid)

plot(theta_grid, exp.(logpdf(mymeasure, theta_grid)), ylim=(0,0.3))

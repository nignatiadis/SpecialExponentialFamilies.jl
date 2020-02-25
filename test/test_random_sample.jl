using ApproxFun
using ExponentialFamilies
using StatsPlots
using Test
using Random

taus = range( -4,  6, step=0.2)
twintower_expfamily = ContinuousExponentialFamilyModel(Uniform(-4, 6), collect(taus); df=8, scale=true)
α_example =  [10; 0.0; 6.0; 9.0; -3.0; -2.0; -8.0; -1.0]
cef_example = twintower_expfamily(α_example)

@test isa(cef_example,  ContinuousExponentialFamily{Uniform{T}} where T<:Real)

@test (ContinuousExponentialFamily{Uniform{T}} where T<:Real) <: Sampleable{Univariate}


Zs = rand(cef_example, 50_000)

x_grid = -4:0.01:6
histogram(Zs, nbins=100, normed=true, color=:lightgrey)
plot!(x_grid,  pdf.(cef_example, x_grid), color=:purple,
          linestyle=:dash, linewidth=3)	

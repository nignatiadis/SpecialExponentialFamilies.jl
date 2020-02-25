# TODO: add proper RNG support via ApproxFun pull request....
function Random.rand!(rng::AbstractRNG, cef::ContinuousExponentialFamily{Uniform{T}} where T<:Real, A::AbstractArray)
	approxfun_interval = ApproxFun.Interval{:closed,:closed}(extrema(cef)...)
	approxfun_fun = ApproxFun.Fun(x->pdf(cef,x, include_base_measure=true), approxfun_interval)
	A[:] = ApproxFun.sample(approxfun_fun, length(A))
end


#function rand(cef::ContinuousExponentialFamily{Uniform{T}})  where T<:Real
#	rand(ce, 1)
#end

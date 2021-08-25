# TODO: add proper RNG support via ApproxFun pull request....
function Random.rand(cef::ExponentialFamilyDistribution, n::Int)
    approxfun_interval = ApproxFun.Interval{:closed,:closed}(extrema(cef)...)
    approxfun_fun = ApproxFun.Fun(
        x -> pdf(cef, x; include_basemeasure=true), approxfun_interval
    )
    return ApproxFun.sample(approxfun_fun, n)
end

#function rand(cef::ContinuousExponentialFamily{Uniform{T}})  where T<:Real
#	rand(ce, 1)
#end

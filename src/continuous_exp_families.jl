struct ExponentialFamily{VF,VS,BM<:Distribution{VF,VS},QF,E} <: Distribution{VF,VS}
    basemeasure::BM
    Q::QF
    integrator::E
end

function ExponentialFamily(basemeasure, Q; integrator=expectation(basemeasure; n=100))
    return ExponentialFamily(basemeasure, Q, integrator)
end

# default to using natural splines
function ExponentialFamily(; basemeasure, df=5, scale=true, kwargs...)
    ns_f = ns_(basemeasure; df=df)
    spline_grid = unique(quantile.(basemeasure, 0:0.001:1.0))

    if scale
        Q1 = ns_f(spline_grid)
        mean_Q1 = mean(Q1; dims=1)
        norm_Q1 = vec(sqrt.(sum(abs2.(Q1 .- mean_Q1); dims=1)))
        mean_Q1 = vec(mean_Q1)
        scaled_ns_f(x) = (ns_f(x) .- mean_Q1) ./ norm_Q1
        cef = ExponentialFamily(basemeasure, scaled_ns_f; kwargs...)
    else
        cef = ExponentialFamily(basemeasure, ns_f; kwargs...)
    end
    return cef
end

function Base.show(io::IO, ef::ExponentialFamily)
    print(io, "Exp. Family")
    print(io, " | base measure = ")
    return print(io, ef.basemeasure)
end

Base.broadcastable(ef::ExponentialFamily) = Ref(ef)
Base.minimum(ef::ExponentialFamily) = Base.minimum(ef.basemeasure)
Base.maximum(ef::ExponentialFamily) = Base.maximum(ef.basemeasure)
Base.extrema(ef::ExponentialFamily) = Base.extrema(ef.basemeasure)

struct ExponentialFamilyDistribution{
    VF,VS,EF<:ExponentialFamily{VF,VS},T<:Real,As<:AbstractVector{T}
} <: Distribution{VF,VS}
    ef::EF
    α::As
    log_normalizing_constant::T
end

function Base.show(io::IO, ef::ExponentialFamilyDistribution)
    println(io, "Exponential Family Distribution")
    println(io, "    | base measure = ", ef.ef.basemeasure)
    return print(io, "      | natural parameters = ", ef.α)
end

function Base.getproperty(exd::ExponentialFamilyDistribution, sym::Symbol)
    if sym in [:basemeasure, :Q, :integrator]
        _prop = Base.getproperty(Base.getfield(exd, :ef), sym)
    else
        _prop = Base.getfield(exd, sym)
    end
    return _prop
end

function _normalizing_constant(ex::ExponentialFamily, α)
    return ex.integrator(x -> exp(dot(ex.Q(x), α)))
end

function ExponentialFamilyDistribution(ex::ExponentialFamily, α)
    norm_const = _normalizing_constant(ex, α)
    return ExponentialFamilyDistribution(ex, α, log(norm_const))
end

function ExponentialFamilyDistribution(basemeasure, Q, α; kwargs...)
    ex = ExponentialFamily(; basemeasure=basemeasure, Q=Q, kwargs...)
    return ExponentialFamilyDistribution(ex, α)
end

# model -> family
function (ex::ExponentialFamily)(α)
    return ExponentialFamilyDistribution(ex, α)
end

# some Base definitions
broadcastable(efd::ExponentialFamilyDistribution) = Ref(efd)
Base.minimum(efd::ExponentialFamilyDistribution) = Base.minimum(efd.basemeasure)
Base.maximum(efd::ExponentialFamilyDistribution) = Base.maximum(efd.basemeasure)
Base.extrema(efd::ExponentialFamilyDistribution) = Base.extrema(efd.basemeasure)

function Distributions.logpdf(
    efd::ExponentialFamilyDistribution, x::Real; include_basemeasure=true
)
    if include_basemeasure
        log_constant = -efd.log_normalizing_constant + logpdf(efd.basemeasure, x)
    else
        log_constant = -efd.log_normalizing_constant
    end
    return dot(efd.Q(x), efd.α) + log_constant
end

function Distributions.pdf(cef::ExponentialFamilyDistribution, x::Real; kwargs...)
    return exp(Distributions.logpdf(cef, x; kwargs...))
end

function Distributions.support(cef::Union{ExponentialFamilyDistribution,ExponentialFamily})
    return support(cef.basemeasure)
end
function Distributions.insupport(
    cef::Union{ExponentialFamilyDistribution,ExponentialFamily}, x::Real
)
    return Distributions.insupport(cef.basemeasure, x)
end

function Distributions.cdf(cef::ExponentialFamilyDistribution, x::Real)
    return error("CDF has not been implemented yet.")
end

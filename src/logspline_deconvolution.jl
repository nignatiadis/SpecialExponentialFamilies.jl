### Fitting Routine through (penalized MLE)

struct EFMarginalDistribution{EFD,EB,S} <: Distribution{Univariate,S}
    efd::EFD
    Z::EB
end

function Base.show(io::IO, ef::EFMarginalDistribution)
    println(io, "MarginalDistribution")
    println(io, " Prior")
    println(io, ef.efd)
    println(io, " EBayes Sample")
    return print(io, ef.Z)
end

function EFMarginalDistribution(
    efd::EFD, Z::EB
) where {EFD,EB<:Empirikos.ContinuousEBayesSample}
    return EFMarginalDistribution{EFD,EB,Continuous}(efd, Z)
end

function EFMarginalDistribution(
    efd::EFD, Z::EB
) where {EFD,EB<:Empirikos.DiscreteEBayesSample}
    return EFMarginalDistribution{EFD,EB,Discrete}(efd, Z)
end

function Distributions.pdf(efm::EFMarginalDistribution, x::Real)
    efd = efm.efd
    Z = efm.Z
    _f =
        μ ->
            pdf(efd, μ; include_basemeasure=false) *
            pdf(Empirikos.likelihood_distribution(Z, μ), x)
    return efd.integrator(_f)
end

function Distributions.logpdf(efm::EFMarginalDistribution, x::Real)
    return log(Distributions.pdf(efm, x))
end

function Distributions.cdf(efm::EFMarginalDistribution, x::Real)
    efd = efm.efd
    Z = efm.Z
    _f =
        μ ->
            pdf(efd, μ; include_basemeasure=false) *
            cdf(Empirikos.likelihood_distribution(Z, μ), x)
    return efd.integrator(_f)
end

function Empirikos.marginalize(Z::EBayesSample, efd::ExponentialFamilyDistribution)
    return EFMarginalDistribution(efd, Z)
end

Base.@kwdef struct PenalizedMLE{EF<:ExponentialFamily,T<:Real}
    ef::EF
    c0::T = 0.01
    solver = NewtonTrustRegion()
    optim_options = Optim.Options(; show_trace=true, show_every=1, g_reltol=1e-8)
    initializer = LindseyMethod(; ef=ef)
end

Base.@kwdef mutable struct FittedPenalizedMLE{P<:PenalizedMLE,T,VT<:AbstractVector{T},EFD}
    pen::P
    α_opt::VT
    α_bias = zero(α_opt)
    efd::EFD = pen.ef(α_opt)
    α_covmat = nothing
    nll_hessian = nothing
    nll_gradient = nothing
    regularizer_hessian = nothing
    regularizer_gradient = nothing
    fitter = nothing
end

Base.broadcastable(fitted::FittedPenalizedMLE) = Ref(fitted)

StatsBase.fit(fitted::FittedPenalizedMLE, args...; kwargs...) = fitted

function StatsBase.fit(pen::PenalizedMLE, Zs::Empirikos.VectorOrSummary)
    return _fit(pen, Zs)
end

function _fit(pen::PenalizedMLE, Zs)
    @unpack ef, c0, solver, optim_options, initializer = pen
    n = nobs(Zs)

    if isa(initializer, LindseyMethod)
        # initialize method through Lindsey's method
        # this mostly makes sense for normal Zs
        fit_init = StatsBase.fit(initializer, response.(Zs), StatsBase.weights(Zs))
        α_init = fit_init.α
    else
        α_init = initializer
    end

    # set up objective
    function _nll(α)
        efd = ef(α)
        return -loglikelihood(Zs, efd) / n
    end
    _s(α) = c0 * norm(α) / n
    _penalized_nll(α) = _nll(α) + _s(α)

    # ready to descend
    optim_res = optimize(_penalized_nll, α_init, solver, optim_options; autodiff=:forward)

    α_opt = Optim.minimizer(optim_res)
    @show α_opt
    #---------------------------------------
    # Extract curvature and bias information
    #---------------------------------------
    hessian_storage_nll = DiffResults.HessianResult(α_opt)
    ForwardDiff.hessian!(hessian_storage_nll, _nll, α_opt)

    nll_hessian = DiffResults.hessian(hessian_storage_nll)
    nll_gradient = DiffResults.gradient(hessian_storage_nll)

    # project onto psd cone
    nll_hessian_eigen = eigen(Symmetric(nll_hessian))
    nll_hessian_eigen.values .= max.(nll_hessian_eigen.values, 0.0)
    nll_hessian = Matrix(nll_hessian_eigen)

    hessian_storage_s = DiffResults.HessianResult(α_opt)
    ForwardDiff.hessian!(hessian_storage_s, _s, α_opt)

    regularizer_hessian = DiffResults.hessian(hessian_storage_s)
    regularizer_gradient = DiffResults.gradient(hessian_storage_s)

    inv_mat = inv((nll_hessian + regularizer_hessian) .* sqrt(n))
    α_covmat = inv_mat * nll_hessian * inv_mat

    α_bias = -inv_mat * regularizer_gradient .* sqrt(n)

    return FittedPenalizedMLE(;
        pen=pen,
        α_opt=α_opt,
        α_covmat=α_covmat,
        α_bias=α_bias,
        nll_hessian=nll_hessian,
        nll_gradient=nll_gradient,
        regularizer_hessian=regularizer_hessian,
        regularizer_gradient=regularizer_gradient,
        fitter=optim_res,
    )
end

function target_bias_std(
    fcef::FittedPenalizedMLE, target::Empirikos.EBayesTarget; bias_corrected=true, clip=true
)
    _fun(α) = target(fcef.pen.ef(α))
    α_opt = fcef.α_opt
    target_gradient = ForwardDiff.gradient(_fun, α_opt)
    target_bias = LinearAlgebra.dot(target_gradient, fcef.α_bias)
    target_variance = LinearAlgebra.dot(target_gradient, fcef.α_covmat, target_gradient)
    target_value = bias_corrected ? _fun(α_opt) - target_bias : _fun(α_opt)

    if clip
        target_value = clamp(target_value, extrema(target)...)
    end

    return (
        estimated_target=target_value,
        estimated_bias=target_bias,
        estimated_std=sqrt(target_variance),
    )
end

function confint(
    fcef::FittedPenalizedMLE,
    target::Empirikos.EBayesTarget;
    level::Real=0.95,
    clip=true,
    bias_corrected=true,
)
    α = 1 - level
    res = target_bias_std(fcef, target; clip=false, bias_corrected=bias_corrected)
    _estimate = res[:estimated_target]
    _std = res[:estimated_std]

    q_mult = quantile(Normal(), 1 - α / 2) * _std

    L, U = _estimate .+ (-1, 1) .* q_mult

    if clip
        L, U = clamp.((L, U), extrema(target)...)
    end

    return Empirikos.BiasVarianceConfidenceInterval(;
        target=target,
        method=nothing,
        α=α,
        se=_std,
        estimate=_estimate,
        lower=L,
        upper=U,
        halflength=q_mult,
    )
end

function confint(
    pmle::Union{PenalizedMLE,FittedPenalizedMLE},
    target::Empirikos.EBayesTarget,
    args...;
    kwargs...,
)
    _fit = StatsBase.fit(pmle, args...)
    return confint(_fit, target; kwargs...)
end

function Base.broadcasted_kwsyntax(
    ::typeof(confint),
    pmle::PenalizedMLE,
    targets::AbstractArray,
    args...;
    level=0.95,
    kwargs...,
)
    _fit = StatsBase.fit(pmle, args...)
    _first_ci = confint(_fit, targets[1]; level=level, kwargs...)
    confint_vec = fill(_first_ci, length(targets))
    for (index, target) in enumerate(targets[2:end])
        confint_vec[index + 1] = StatsBase.confint(_fit, target; level=level)
    end
    return confint_vec
end

function Base.broadcasted(::typeof(confint), pmle::PenalizedMLE, targets, args...)
    return confint.(pmle, targets, args...; level=0.95)
end

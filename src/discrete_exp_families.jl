mutable struct DiscreteExponentialFamilyModel{T<:Real,
                                              Ts<:AbstractVector{T},
                                              Qs<:AbstractMatrix{T}}
    support::Ts
    Q::Qs
end

function DiscreteExponentialFamilyModel(support, suffstats::Function; scale=true)
    Q = vcat(suffstats.(support)...)
    if scale
        tr = fit(ZScoreTransform, Q', center=true, scale=true)
        Q = copy(transpose(StatsBase.transform(tr, Q')))
    end
    DiscreteExponentialFamilyModel(support, Q)
end

function probs(dem::DiscreteExponentialFamilyModel, α)
    pr = exp.(dem.Q*α)
    pr ./ sum(pr)
end

struct DiscreteExponentialFamily{T<:Real,
                                 Ts<:AbstractVector{T},
                                 Qs<:AbstractMatrix{T},
                                 As<:AbstractVector{T},
                                 D<:DiscreteNonParametric} <: DiscreteUnivariateDistribution
    support::Ts
    Q::Qs
    α::As
    dbn::D
end

function DiscreteExponentialFamily(dem::DiscreteExponentialFamilyModel, α)
    ps = probs(dem, α)
    dbn = DiscreteNonParametric(dem.support, ps)
    DiscreteExponentialFamily(dem.support,
                              dem.Q,
                              α,
                              dbn)
end


Base.eltype(::Type{<:DiscreteExponentialFamilyModel{T}}) where T = T
Base.eltype(::Type{<:DiscreteExponentialFamily{T}}) where T = T

function StatsBase.moment(dem::DiscreteExponentialFamilyModel, α, r)
    ts = support(dem)
    ps = probs(dem, α)
    dot(ps, ts.^r)
end

function StatsBase.moment(dem::DiscreteExponentialFamily, r)
    ts = support(dem)
    ps = probs(dem)
    dot(ps, ts.^r)
end

StatsModels.@delegate DiscreteExponentialFamily.dbn [probs, pdf, mean, var, cf, mgf]

function rand(rng::AbstractRNG, dem::DiscreteExponentialFamily)
    rand(rng, dem.dbn)
end

sampler(dem::DiscreteExponentialFamily) = sampler(dem.dbn)


support(dem::DiscreteExponentialFamily) = dem.support
support(dem::DiscreteExponentialFamilyModel) = dem.support

broadcastable(dem::DiscreteExponentialFamily) = Ref(dem)
broadcastable(dem::DiscreteExponentialFamilyModel) = Ref(dem)

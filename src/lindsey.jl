
Base.@kwdef mutable struct LindseyMethod{EF, ST}
    ef::EF
    discretizer::ST = Empirikos.Discretizer()
 end


 function StatsBase.fit(lindsey::LindseyMethod, Xs::AbstractVector{<:Real}, args...)
    lindsey = Empirikos.set_defaults(lindsey, Xs)
    _fit(lindsey, Xs, args...)
 end

 function StatsBase.fit(lindsey::LindseyMethod, Xs, args...)
    _fit(lindsey, Xs, args...)
 end

 function _fit(lindsey::LindseyMethod, Xs::AbstractVector{<:Real}, args...)
    Xs_summary = summarize(Xs, args..., lindsey.discretizer)
    _fit(lindsey, Xs_summary)
 end

 function _fit(lindsey::LindseyMethod, Xs::AbstractVector{<:Interval}, args...)
    Xs_summary = summarize(Xs, args...)
    _fit(lindsey, Xs_summary)
 end

function _fit(lindsey::LindseyMethod, Xs::Empirikos.MultinomialSummary)
    mdpts = (first.(keys(Xs)) .+ last.(keys(Xs))) ./ 2

    ef = lindsey.ef

     #poisson_predictor = cefm.Q.(collect(mdpts))
    poisson_predictor = vcat(ef.Q.(mdpts)'...)
    poisson_predictor = hcat( fill(1.0, length(mdpts)), poisson_predictor)
    poisson_response = values(StatsBase.weights(Xs))
    poisson_offset = pdf.(ef.basemeasure, mdpts)

    poisson_fit = fit(GeneralizedLinearModel, poisson_predictor, poisson_response,
                            Poisson(); offset=poisson_offset)

    αs = coef(poisson_fit)[2:end]
    ef(αs)
 end

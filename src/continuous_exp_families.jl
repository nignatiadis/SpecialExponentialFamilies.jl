struct ContinuousExponentialFamilyModel{BM<:Distribution,
                                   QF<:Function}
    base_measure::BM
    Q::QF
    Q_dim::Int
end


function ContinuousExponentialFamilyModel(base_measure, Q)
   example_eval = Q([0.0])
   Q_dim = size(example_eval, 2)
   ContinuousExponentialFamilyModel(base_measure, Q, Q_dim)
end


Base.minimum(cefm::ContinuousExponentialFamilyModel) = Base.minimum(cefm.base_measure)
Base.maximum(cefm::ContinuousExponentialFamilyModel) = Base.maximum(cefm.base_measure)
Base.extrema(cefm::ContinuousExponentialFamilyModel) = Base.extrema(cefm.base_measure)

struct ContinuousExponentialFamily{BM<:Distribution,
                                   QF<:Function,
                                   T<:Real,
                                   As<:AbstractVector{T}}
    base_measure::BM
    Q::QF
    α::As
    log_normalizing_constant::T
end


_default_integrator(base_measure, n_points) = expectation(base_measure; n=n_points)


function _normalizing_constant(base_measure, Q, α;
                      n_points = 100,
                      integrator = _default_integrator(base_measure, n_points))
   integrator(x -> exp(dot(Q([x]), α)))
end

function ContinuousExponentialFamily(base_measure, Q, α; kwargs...)
   norm_const = _normalizing_constant(base_measure, Q, α; kwargs...)
   ContinuousExponentialFamily(base_measure, Q, α, log(norm_const))
end


# some notational clash here for now.
function logpdf(cef::ContinuousExponentialFamily, x)
  cef.Q(x)*cef.α .+ cef.log_normalizing_constant .+ logpdf.(cef.base_measure, x)
end

pdf(cef::ContinuousExponentialFamily, x) = exp.(logpdf(cef, x))


struct LindseyMethod
   grid
end


function fit(cefm::ContinuousExponentialFamilyModel, Xs; kwargs...)
   fit(cefm, Xs, LindseyMethod; kwargs... )
end

function fit(cefm::ContinuousExponentialFamilyModel, Xs, ::Type{LindseyMethod}; length=100)
   model_a, model_b = extrema(cefm)
   data_a, data_b = extrema(Xs) .+ [-0.01; 0.01]
   lindsey_grid = range( max(model_a, data_a), min(model_b, data_b); length=length)
   fit(cefm, Xs, LindseyMethod(lindsey_grid))
end

function fit(cefm::ContinuousExponentialFamilyModel, Xs, ls::LindseyMethod)
   hist = fit(Histogram, Xs, ls.grid)
   mdpts = StatsBase.midpoints(hist.edges[1])

   poisson_predictor = cefm.Q(collect(mdpts))
   poisson_predictor = hcat( fill(1.0, length(mdpts)), poisson_predictor)

   poisson_offset = pdf.(cefm.base_measure, mdpts)

   poisson_fit = fit(GeneralizedLinearModel, poisson_predictor, hist.weights,
                           Poisson(); offset=poisson_offset)

   αs = coef(poisson_fit)[2:end]
   ContinuousExponentialFamily(cefm.base_measure, cefm.Q, αs)
end

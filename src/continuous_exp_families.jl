struct ContinuousExponentialFamilyModel{BM<:Distribution,
                                   QF}
    base_measure::BM
    Q::QF
    Q_dim::Int
end


function ContinuousExponentialFamilyModel(base_measure, Q)
   example_eval = Q([0.0])
   Q_dim = size(example_eval, 2)
   ContinuousExponentialFamilyModel(base_measure, Q, Q_dim)
end

# default to using natural splines
function ContinuousExponentialFamilyModel(base_measure, spline_grid::AbstractVector{<:Real}; df=5, scale=true )

   ns_f = ns_(spline_grid; df=df)

   if scale
      Q1 = ns_f(spline_grid)
      mean_Q1 = mean(Q1; dims=1)
      Q1 = Q1 .- mean_Q1
      norm_Q1 = vec(sqrt.(sum( abs2.(Q1); dims=1)))
      mean_Q1 = vec(mean_Q1)
      scaled_ns_f(x) = (ns_f(x) .- mean_Q1)./norm_Q1
      cef = ContinuousExponentialFamilyModel(base_measure, scaled_ns_f)
   else
      cef = ContinuousExponentialFamilyModel(base_measure, ns_f)
   end
   cef
end

broadcastable(cefm::ContinuousExponentialFamilyModel) = Ref(cefm)
Base.minimum(cefm::ContinuousExponentialFamilyModel) = Base.minimum(cefm.base_measure)
Base.maximum(cefm::ContinuousExponentialFamilyModel) = Base.maximum(cefm.base_measure)
Base.extrema(cefm::ContinuousExponentialFamilyModel) = Base.extrema(cefm.base_measure)

struct ContinuousExponentialFamily{BM<:Distribution,
                                   QF, #should be callable
                                   T<:Real,
                                   As<:AbstractVector{T}} <: Distribution{Univariate, Continuous}
    base_measure::BM
    Q::QF
    α::As
    log_normalizing_constant::T
end


_default_integrator(base_measure, n_points) = expectation(base_measure; n=n_points)


function _normalizing_constant(base_measure, Q, α;
                      n_points = 100,
                      integrator = _default_integrator(base_measure, n_points))
   integrator(x -> exp(dot(Q(x), α)))
end

function ContinuousExponentialFamily(base_measure, Q, α; kwargs...)
   norm_const = _normalizing_constant(base_measure, Q, α; kwargs...)
   ContinuousExponentialFamily(base_measure, Q, α, log(norm_const))
end


# model -> family
function (cefm::ContinuousExponentialFamilyModel)(α; kwargs...)
	ContinuousExponentialFamily(cefm.base_measure, cefm.Q, α; kwargs...)
end



# some Base definitions
broadcastable(cef::ContinuousExponentialFamily) = Ref(cef)
Base.minimum(cef::ContinuousExponentialFamily) = Base.minimum(cef.base_measure)
Base.maximum(cef::ContinuousExponentialFamily) = Base.maximum(cef.base_measure)
Base.extrema(cef::ContinuousExponentialFamily) = Base.extrema(cef.base_measure)



# some notational clash here for now.
function logpdf(cef::ContinuousExponentialFamily, x; include_base_measure = false)
  if include_base_measure
     log_constant =  -cef.log_normalizing_constant + logpdf(cef.base_measure, x)
  else
     log_constant = -cef.log_normalizing_constant
  end
  dot(cef.Q(x),cef.α) + log_constant
end

pdf(cef::ContinuousExponentialFamily, x; kwargs...) = exp(logpdf(cef, x; kwargs...))

support(cef::Union{ContinuousExponentialFamily, ContinuousExponentialFamilyModel}) = support(cef.base_measure)
insupport(cef::Union{ContinuousExponentialFamily, ContinuousExponentialFamilyModel}, x) = insupport(cef.base_measure, x)

mutable struct LindseyMethod{ST}
   grid::ST
	nedges::Int
end

LindseyMethod(grid::AbstractVector) = LindseyMethod(grid, length(grid))
LindseyMethod(nedges::Int) = LindseyMethod(nothing, nedges)
LindseyMethod() = LindseyMethod(500)



function fit(cefm::ContinuousExponentialFamilyModel, Xs::AbstractVector)
   fit(cefm, Xs, LindseyMethod)
end

function fit(cefm::ContinuousExponentialFamilyModel, Xs::AbstractVector, ls::LindseyMethod{Nothing})
   model_a, model_b = extrema(cefm)
   data_a, data_b = extrema(Xs) .+ [-0.01; 0.01]
   lindsey_grid = range( max(model_a, data_a), min(model_b, data_b); length=ls.nedges)
	ls_updated = LindseyMethod(lindsey_grid)
   fit(cefm, Xs, ls_updated)
end


function fit(cefm::ContinuousExponentialFamilyModel, Xs::AbstractVector, ls::LindseyMethod{<:AbstractVector})
   hist = fit(Histogram, Xs, ls.grid)
	fit(cefm, hist, ls)
end

function fit(cefm::ContinuousExponentialFamilyModel, hist::Histogram, ls::LindseyMethod)
   mdpts = StatsBase.midpoints(hist.edges[1])

	keep_idx = insupport.(cefm, mdpts)
	mdpts = mdpts[keep_idx]
   #poisson_predictor = cefm.Q.(collect(mdpts))
	poisson_predictor = vcat(cefm.Q.(mdpts)'...)
   poisson_predictor = hcat( fill(1.0, length(mdpts)), poisson_predictor)

   poisson_offset = pdf.(cefm.base_measure, mdpts)
   poisson_fit = fit(GeneralizedLinearModel, poisson_predictor, hist.weights,
                           Poisson(); offset=poisson_offset)

   αs = coef(poisson_fit)[2:end]
   ContinuousExponentialFamily(cefm.base_measure, cefm.Q, αs)
end

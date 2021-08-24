using Revise
using Plots
using Optim
using ExponentialFamilies
using Combinatorics
using IterTools
using RCall
using Random
using LaTeXStrings

support_grid = -3:0.2:3



EfronTwoTowers = MixtureModel( [Uniform(-1.7, -0.7),
                                Uniform(0.7, 2.7)],
                                [1/3, 2/3])


dem_poly = DiscreteExponentialFamilyModel(support_grid, x->[x x^2 x^3])

tmp_probs = probs(dem_poly, [1.0; -1.0; 0.0])

plot(support_grid,tmp_probs)

tau = -3:0.05:3
@rput tau
R"library(splines)"
R"""Q1 <- scale(splines::ns(tau, 5), center = TRUE, scale = FALSE)
    Q1 <- apply(Q1, 2, function(w) w / sqrt(sum(w * w)))
 """
@rget Q1

dem_spline = DiscreteExponentialFamilyModel(tau, Q1)
d1 = DiscreteExponentialFamily(dem_poly, [1.0; -1.0; 0.0])

d2 = DiscreteExponentialFamily(dem_spline, [1.0; -1.0; 0.0; 1.0; 1.0])

plot(support(d2), probs(d2), seriestype=:sticks)
Zs = rand(d1,1000)

function deconvolved_empirical_moment(Zs, r)
    u_stats = Vector{Float64}(undef, size(Zs,1))
    for (i, row) in enumerate(eachrow(Zs))
        u_stats[i] = mean([prod(comb) for comb in combinations(row,r)])
    end
    mean(u_stats)
end

deconvolved_empirical_moments(Zs, rs) =  deconvolved_empirical_moment.(Ref(Zs), rs)

function empirical_moment(Zs, r::Integer)
    mean(Zs.^r)
end

empirical_moments(Zs, rs) =  empirical_moment.(Ref(Zs), rs)


function deconvolve_moments(dem::DiscreteExponentialFamilyModel, Zs, R; deconv=true)

    if deconv
        m_hat = deconvolved_empirical_moments(Zs,1:R)
    else
        m_hat = empirical_moments(Zs,1:R)
    end

    function my_loss(α)
        moments_α = [moment(dem, α, r) for r in 1:R]
        sum( (moments_α .- m_hat).^2)
    end

    α_init = ones(eltype(dem), size(dem.Q, 2))

    optim_tst = optimize(my_loss, α_init)
    min1 = Optim.minimizer(optim_tst)
    DiscreteExponentialFamily(dem, min1)
end




Random.seed!(1)
n=50_000
μs=rand(MarronWandSkewedBimodal, n)
εs = rand(Normal(0,1),n, 10).*sqrt(10)
Zs = εs .+ μs
Zs_mean = vec(mean(Zs; dims=2))

tmp = deconvolve_moments(dem_spline, Zs, 5)

@rput Zs_mean

R"""
 library(deconvolveR)
 deconv_res <- deconv(tau, Zs_mean, family='Normal')
 a_g_hat <- deconv_res$mle
 deconv_res$mle
 """

@rget a_g_hat
efron_g = DiscreteExponentialFamily(dem_spline, a_g_hat)

pgfplots()
plot_1 = plot(support(tmp), [probs(tmp)./0.05 probs(efron_g)./0.05 pdf.(MarronWandSkewedBimodal,
                                    support(tmp))],
                                    color=["purple" "orange" "black"],
                                    width=[2 2 0.7],
                                    alpha=[1 1 0.7],
                                    linestyle=[:dot :dash :solid],
                                    ylab="Density",
                                    label=["Deconvolved Moments" "Oracle MLE" "Truth"],
                                    legend=:topleft,
                                    xlab=L"\mu")



scale_lapl = rand(Uniform(0,sqrt(5)), n)
εs_lapl = rand(Laplace(0,1),n, 10).*scale_lapl
Zs_lapl = εs_lapl .+ μs
Zs_lapl_mean = vec(mean(Zs; dims=2))


tmp_lapl = deconvolve_moments(dem_spline, Zs_lapl, 5)

@rput Zs_lapl_mean

R"""
 library(deconvolveR)
 deconv_res2 <- deconv(tau, Zs_lapl_mean, family='Normal')
 a_g_hat2 <- deconv_res2$mle
 """

@rget a_g_hat2
efron_g_lapl = DiscreteExponentialFamily(dem_spline, a_g_hat2)


pgfplots()
plot_2 = plot(support(tmp_lapl), [probs(tmp_lapl)./0.05 pdf.(MarronWandSkewedBimodal,
                                    support(tmp_lapl))],
                                    color=["purple"  "black"],
                                    width=[2  0.7],
                                    alpha=[1  0.7],
                                    linestyle=[:dot :solid],
                                    ylab="Density",
                                    label=["Deconvolved Moments" "Truth"],
                                    legend=:topleft,
                                    xlab=L"\mu")


upscale = 0.6#8 #8x upscaling in resolution
default(size=(2000*upscale,400*upscale))
pl = plot(plot_1, plot_2, title=["A" "B"],
            fglegend = :transparent,
            grid = false,
            size=(1500*upscale,600*upscale))
savefig(pl,"deconv_plots2.pdf")
moment_kernel.(cmbs)
moment_kernel(xs) = prod(xs)



fit(dem::DiscreteExponentialFamilyModel, R::Integer, Zs)

R = 3
m_hat = empirical_moments(Zs, 1:3)

my_loss([1.0;-1.0;0.0])

moment(dem_poly, 2, α_init)
moment(d1 )

α_init = ones(eltype(d1), size(d1.Q, 2))


min2 = Optim.minimizer(optim_tst)


moment.(d1, 1:5)

probs(d1.dbn)

moment(dem, [1.0,-1.0,1.0])

using SpecialExponentialFamilies
using Documenter

makedocs(;
    modules=[SpecialExponentialFamilies],
    authors="Nikos Ignatiadis <nikos.ignatiadis01@gmail.com> and contributors",
    repo="https://github.com/nignatiadis/SpecialExponentialFamilies.jl/blob/{commit}{path}#L{line}",
    sitename="ExponentialFamilies.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nignatiadis.github.io/SpecialExponentialFamilies.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

#deploydocs(;
#    repo="github.com/nignatiadis/SpecialExponentialFamilies.jl",
#)

using Documenter, ExponentialFamilies

makedocs(;
    modules=[ExponentialFamilies],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/nignatiadis/ExponentialFamilies.jl/blob/{commit}{path}#L{line}",
    sitename="ExponentialFamilies.jl",
    authors="Nikos Ignatiadis <nikos.ignatiadis01@gmail.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/nignatiadis/ExponentialFamilies.jl",
)

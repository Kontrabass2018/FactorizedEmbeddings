using Documenter, FactorizedEmbeddings
push!(LOAD_PATH, "../src/")
makedocs(sitename="Factorized Embeddings Documentation",
pages = ["Home"=>"index.md",
"Guide" => "guide.md",
"Public Interface" => "pub_interface.md"])
deploydocs(; repo="https://github.com/Kontrabass2018/FactorizedEmbeddings")
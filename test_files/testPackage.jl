include("functions.jl")
#FE related
# Pkg.add("https://github.com/Kontrabass2018/FactorizedEmbeddings")
using FactorizedEmbeddings

TCGA_data, labs, samples, genes, biotypes = fetch_data("TCGA_TPM_lab.h5", shfl = true)
TCGA_data
model_params = generate_params(TCGA_data[:,biotypes .== "protein_coding"], nsteps_dim_redux = 10_000)
using Flux
using CUDA
using Random

model = FactorizedEmbeddings.fit(TCGA_data[:,biotypes .== "protein_coding"], model_params)


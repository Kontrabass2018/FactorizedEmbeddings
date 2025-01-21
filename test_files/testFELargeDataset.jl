## load TCGA data 
## Create fake data by duplicating each example.
cd("test_files")
include("functions.jl")
#FE related
using FactorizedEmbeddings
device!(3)

TCGA_data, labs, samples, genes, biotypes = fetch_data("TCGA_TPM_lab.h5", shfl = true);
cds_data = TCGA_data[:, biotypes .== "protein_coding"]
large_data = vcat([cds_data for i in 1:10]...)

size(large_data)

model = fit(large_data, generate_params(large_data, nsteps_dim_redux = 40_000, l2_val =1e-8, emb_size_2 = 100, fe_layers_size = [100,100,100]), verbose = 1)

large_embed = cpu(model[1][1].weight)
large_labs = vcat(labs, labs)

TCGA_colors_file = "TCGA_colors_def.txt"
fig = Figure(size = (1024,800));
ax = Axis(fig[1,1],title="Trained 2x TCGA with FE during 40,000 steps", xlabel = "Patient-Embed-1", ylabel="Patient-Embed-2", aspect = 1);
colors_labels_df = CSV.read(TCGA_colors_file,  DataFrame)
# first plot train embed with circles.
for (i, group_lab) in enumerate(unique(labs))
    group = large_labs .== group_lab
    col = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"hexcolor"][1]
    name = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"name"][1]
    scatter!(ax, large_embed[1,group], large_embed[2,group], strokewidth = 0.1, color = String(col), label = name, marker = :circle)
end 
fig

#### With resampling!

infile = "/home/golem/scratch/munozc/DDPM/full_TCGA_GE.h5"
inf = h5open(infile, "r")
TCGA_data = inf["data_matrix"][:,:]
biotypes = inf["gene_type"][:]
cancer_types = inf["cancer_type"][:]
close(inf)

nreplicates = 3 
BIG_MATRIX = vec(TCGA_data) * ones(nreplicates)'
@time SAMPLED_MATRIX = rand.(Poisson.(reshape(BIG_MATRIX', (size(TCGA_data)[1] * nreplicates,size(TCGA_data)[2]))))
cov = sum(SAMPLED_MATRIX, dims = 2)
SAMPLED_DATA = SAMPLED_MATRIX * 1e6 ./ cov
large_data = log10.(SAMPLED_DATA .+ 1)

model = FactorizedEmbeddings.fit(large_data, generate_params(large_data, nsteps_dim_redux = 40_000, l2_val =1e-8, emb_size_2 = 100, fe_layers_size = [100,100,100]), verbose = 1)

large_embed = cpu(model[1][1].weight)
large_labs = vec(permutedims(reshape(repeat(cancer_types,nreplicates), (size(TCGA_data)[1],nreplicates))))
TCGA_colors_file = "TCGA_colors_def.txt"
fig = Figure(size = (1024,800));
ax = Axis(fig[1,1],title="Trained 2x TCGA with FE during 40,000 steps", xlabel = "Patient-Embed-1", ylabel="Patient-Embed-2", aspect = 1);
colors_labels_df = CSV.read(TCGA_colors_file,  DataFrame)
# first plot train embed with circles.
for (i, group_lab) in enumerate(unique(large_labs))
    group = large_labs .== group_lab
    col = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"hexcolor"][1]
    name = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"name"][1]
    scatter!(ax, large_embed[1,group], large_embed[2,group], strokewidth = 0.1, color = String(col), label = name, marker = :circle)
end 
fig

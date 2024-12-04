cd("test_files")
include("functions.jl")
#FE related
# Pkg.add("https://github.com/Kontrabass2018/FactorizedEmbeddings")
using FactorizedEmbeddings


TCGA_data, labs, samples, genes, biotypes = fetch_data("TCGA_TPM_lab.h5", shfl = true);
# model_params = generate_params(TCGA_data[:,biotypes .== "protein_coding"], nsteps_dim_redux = 10_000)
# @time model = FactorizedEmbeddings.fit(TCGA_data[:,biotypes .== "protein_coding"], model_params)


folds = split_train_test(TCGA_data);

train_data = folds[1]["train_x"];
test_data = folds[1]["test_x"];
model_params = generate_params(train_data, nsteps_dim_redux = 100_000, nsamples_batchsize=1)
model = FactorizedEmbeddings.fit(train_data, model_params;verbose = 1);
# infer
infer_model, model_phase_1 = FactorizedEmbeddings.infer(model, train_data, test_data, model_params, verbose = 1)
# plot 

train_embed = cpu(model[1][1].weight)
test_embed = model_phase_1

train_ids = folds[1]["train_ids"]
test_ids = folds[1]["test_ids"]

TCGA_colors_file = "TCGA_colors_def.txt"
fig = Figure(size = (1024,800));
ax = Axis(fig[1,1],title="Train and Test patient embedding", xlabel = "Patient-FE-1", ylabel="Patient-FE-2", aspect = 1);
colors_labels_df = CSV.read(TCGA_colors_file,  DataFrame)
# first plot train embed with circles.
for (i, group_lab) in enumerate(unique(labs))
    group = labs[train_ids] .== group_lab
    col = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"hexcolor"][1]
    name = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"name"][1]
    scatter!(ax, train_embed[1,group], train_embed[2,group], strokewidth = 0.1, color = String(col), label = name, marker = :circle)
end 
for (i, group_lab) in enumerate(unique(labs))
    group = labs[test_ids] .== group_lab
    col = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"hexcolor"][1]
    name = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"name"][1]
    scatter!(ax, test_embed[1,group], test_embed[2,group], strokewidth = 1, color = String(col), label = name, marker = :utriangle)
end 
fig

test_embed = cpu(infer_model[1][1].weight);
fig = Figure(size = (1024,800));
ax = Axis(fig[1,1],title="Train and Test patient embedding", xlabel = "Patient-FE-1", ylabel="Patient-FE-2", aspect = 1);
colors_labels_df = CSV.read(TCGA_colors_file,  DataFrame)
# first plot train embed with circles.
for (i, group_lab) in enumerate(unique(labs))
    group = labs[train_ids] .== group_lab
    col = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"hexcolor"][1]
    name = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"name"][1]
    scatter!(ax, train_embed[1,group], train_embed[2,group], strokewidth = 0.1, color = String(col), label = name, marker = :circle)
end 
for (i, group_lab) in enumerate(unique(labs))
    group = labs[test_ids] .== group_lab
    col = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"hexcolor"][1]
    name = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"name"][1]
    scatter!(ax, test_embed[1,group], test_embed[2,group], strokewidth = 1, color = String(col), label = name, marker = :utriangle)
end 
fig




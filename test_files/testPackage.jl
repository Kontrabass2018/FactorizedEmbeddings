cd("test_files")
include("functions.jl")
#FE related
# Pkg.add("https://github.com/Kontrabass2018/FactorizedEmbeddings")
using FactorizedEmbeddings
device!(3)

TCGA_data, labs, samples, genes, biotypes = fetch_data("TCGA_TPM_lab.h5", shfl = true);
folds = split_train_test(TCGA_data);

train_data = folds[1]["train_x"];
test_data = folds[1]["test_x"];

model_params = generate_params(train_data, nsteps_dim_redux = 300_000, l2_val =1e-8, emb_size_2 = 100, fe_layers_size = [100,100,100])
# @time model = FactorizedEmbeddings.fit(TCGA_data[:,biotypes .== "protein_coding"], model_params)




# train
model = FactorizedEmbeddings.fit(train_data, model_params)

include("testUtils.jl")
new_embed = infer_dev(model, train_data, test_data, model_params, verbose = 1)

X_test, Y_test = prep_FE(test_data;order = "per_sample", verbose = 1);
batchsize = 100_000
nminibatches = Int(floor(length(Y_test) / batchsize))
    
inference_model = init_inference_FE(model, new_embed)
# opt = OptimiserChain(WeightDecay(1e-7), Adam(1e-4))
opt = Adam(1e-2)
nsteps = 10_000
state = Flux.setup(opt, inference_model) |> gpu 
p = Progress(nsteps; showspeed=true)   
for iter in 1:nsteps
    cursor = (iter -1)  % nminibatches + 1
    btch_rng = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, length(Y_test)))
    
    X_ = (X_test[1][btch_rng],X_test[2][btch_rng])
    Y_ = Y_test[btch_rng]

    grads = Flux.gradient(inference_model) do m 
        Flux.mse(m(X_), Y_)
    end
    lossval = Flux.mse(inference_model(X_), Y_) + 1e-7 * sum(sum.(abs2, Flux.trainables(inference_model)))
    pearson = my_cor(inference_model(X_), Y_)
    Flux.update!(state, inference_model, grads[1])
    next!(p; showvalues=[(:step,iter), (:loss,lossval),(:pearson, pearson)])
end 
# infer
# infer_model, model_phase_1 = FactorizedEmbeddings.infer(model, train_data, test_data, model_params, verbose = 1)
# include("testUtils.jl")

# plot results 

train_embed = cpu(model[1][1].weight)
test_embed = cpu(inference_model.embed.emb1.weight)

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
fig
for (i, group_lab) in enumerate(unique(labs))
    group = labs[test_ids] .== group_lab
    col = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"hexcolor"][1]
    name = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"name"][1]
    scatter!(ax, test_embed[1,group], test_embed[2,group], strokewidth = 1, color = String(col), label = name, marker = :utriangle)
end 
fig

test_embed = cpu(inference_model.embed.emb1.weight)
model_phase1 = new_embed
fig = Figure(size = (1024,800));
ax = Axis(fig[1,1],title="Train and Test patient embedding", xlabel = "Patient-FE-1", ylabel="Patient-FE-2", aspect = 1);
for (i, group_lab) in enumerate(unique(labs))
    group = labs[test_ids] .== group_lab
    col = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"hexcolor"][1]
    name = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"name"][1]
    scatter!(ax, model_phase1[1,group], model_phase1[2,group], strokewidth = 0.1, color = String(col), label = name, marker = :circle)
end 
fig
for (i, group_lab) in enumerate(unique(labs))
    group = labs[test_ids] .== group_lab
    col = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"hexcolor"][1]
    name = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"name"][1]
    scatter!(ax, test_embed[1,group], test_embed[2,group], strokewidth = 1, color = String(col), label = name, marker = :utriangle)
end 
fig
CairoMakie.save("../figures/test_inference_300_000.svg", fig)
CairoMakie.save("../figures/test_inference_300_000.png", fig)


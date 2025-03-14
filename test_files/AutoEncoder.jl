cd("test_files")
include("functions.jl")
using ProgressMeter
function my_cor(X::AbstractVector, Y::AbstractVector)
    sigma_X = std(X)
    sigma_Y = std(Y)
    mean_X = mean(X)
    mean_Y = mean(Y)
    cov = sum((X .- mean_X) .* (Y .- mean_Y)) / length(X)
    return cov / sigma_X / sigma_Y
end 
## load in data

TCGA_data, labs, samples, genes, biotypes = fetch_data("TCGA_TPM_lab.h5", shfl = true);

## define Auto-Encoder
nsamples, ngenes = size(TCGA_data)
nsteps, lr, l2 = 100_000, 1e-4, 1e-5
AE = Chain(Dense(ngenes, 100, relu),
Dense(100,100, relu),
Dense(100, 2,identity), 
Dense(2,100, relu),
Dense(100,100, relu),
Dense(100,ngenes, relu)) |> gpu

X = Matrix(TCGA_data') |> gpu
batchsize = 200
nminibatches = Int(floor(nsamples / batchsize))
opt = Flux.Adam(lr)
state = Flux.setup(opt, AE) |> gpu 
p = Progress(nsteps; showspeed = true)
for iter in 1:nsteps
    cursor = (iter -1)  % nminibatches + 1
    batch_range = (cursor -1) * batchsize + 1 : cursor * batchsize
    X_ = X[:,batch_range]
    grads = Flux.gradient(AE) do m
        Flux.mse(m(X_), X_) + l2 * sum(sum.(abs2, Flux.trainables(m))) 
    end
    outs = AE(X_)
    lossval = Flux.mse(outs, X_) + l2 * sum(sum.(abs2, Flux.trainables(AE)))      
    pearson = my_cor(vec(outs), vec(X_))
    Flux.update!(state, AE, grads[1])
    next!(p; showvalues=[(:step,iter), (:loss, lossval), (:pearson, pearson)])
end 

X_tr = Matrix(cpu(AE[3](AE[2](AE[1](X)))))
TCGA_colors_file = "TCGA_colors_def.txt"
fig = Figure(size = (1024,800));
ax = Axis(fig[1,1],title="Reductions with an Auto-Encoder", xlabel = "Dimension 1", ylabel="Dimension 2", aspect = 1);
colors_labels_df = CSV.read(TCGA_colors_file,  DataFrame)
for (i, group_lab) in enumerate(unique(labs))
    group = labs.== group_lab
    col = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"hexcolor"][1]
    name = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"name"][1]
    scatter!(ax, X_tr[1,group], X_tr[2,group], strokewidth = 0.1, color = String(col), label = name, marker = :circle)
end 
fig
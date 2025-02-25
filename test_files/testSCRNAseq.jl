## get file directories, load matrices.
cd("test_files")
using Pkg
Pkg.activate(".")
include("functions.jl")
using FactorizedEmbeddings

generate_params_FE(nsamples::Int, ngenes::Int;    
emb_size::Int=2, emb_size_2::Int=100, 
nsteps_dim_redux::Int=1000, l2_val::Float64=1e-7,
batchsize::Int=20_000, 
fe_layers_size = [100, 50, 50]
) = return Dict( 
## run infos 
# "session_id" => session_id,  "modelid" =>  "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
# "outpath"=>outpath, 
"machine_id"=>strip(read(`hostname`, String)), # "device" => "$(CUDA.device())", 
## data infos 
"nsamples" =>nsamples, "ngenes"=> ngenes,  
## optim infos 
"lr" => 5e-3, "l2" =>l2_val,"nsteps" => nsteps_dim_redux, "nsteps_inference" => Int(floor(nsteps_dim_redux * 0.1)), "batchsize" => batchsize,
## model infos
"emb_size_1" => emb_size, "emb_size_2" => emb_size_2, "fe_layers_size"=> fe_layers_size,
)

function my_cor(X::AbstractVector, Y::AbstractVector)
    sigma_X = std(X)
    sigma_Y = std(Y)
    mean_X = mean(X)
    mean_Y = mean(Y)
    cov = sum((X .- mean_X) .* (Y .- mean_Y)) / length(X)
    return cov / sigma_X / sigma_Y
end 

using ProgressMeter
function train_dev!(params, X, Y, model;verbose = 0)
    verbose > 0 && (println("Training model..."))
    batchsize = params["batchsize"] 
    nminibatches = Int(floor(size(X[1])[1] / batchsize))
    opt = Flux.Adam(params["lr"])
    state = Flux.setup(opt, model) |> gpu 
    p = Progress(params["nsteps"]; showspeed=true)
    for iter in 1:params["nsteps"]
        # Stochastic gradient descent with minibatches
        cursor = (iter -1)  % nminibatches + 1
        
        batch_range = (cursor -1) * batchsize + 1 : cursor * batchsize
        X_, Y_ = (X[1][batch_range],X[2][batch_range]), Y[batch_range] # Access via "view" : quick   
        grads = Flux.gradient(model) do m 
            Flux.mse(m(X_), Y_) + params["l2"] * sum(sum.(abs2, Flux.trainables(m)))  ## loss
        end
        lossval = Flux.mse(model(X_), Y_) + params["l2"] * sum(sum.(abs2, Flux.trainables(model))) 
        pearson = my_cor(model(X_), Y_)
        Flux.update!(state, model, grads[1])
        # println("FE $(iter) epoch $(Int(ceil(iter / nminibatches))) - $cursor /$nminibatches - TRAIN loss: $(lossval)\tpearson r: $pearson ELAPSED: $((now() - start_timer).value / 1000 )") : nothing         
        next!(p; showvalues=[(:step,iter), (:loss, lossval), (:pearson, pearson)])
    end
    return model 
end 

## function that returns X, Y (X sample id is offshifted by N given by user)
function get_coord_mat(fpath)
    dat = CSV.read(fpath, DataFrame, skipto=4, header = false)
    X_sample = dat[:,2] 
    X_gene = dat[:,1]
    Y = dat[:,3]
    return dat, X_sample, X_gene, Y
end 


offs, Xs, Xg, Y, conditions, datasets = 0, [], [], [], [], []
for (fid, cond) in enumerate(readdir("../Data/scRNAseq/"))
    fpath = "../Data/scRNAseq/$cond/matrix.mtx"
    df, condXs, condXg, condY = get_coord_mat(fpath)
    nsamples = maximum(condXs)
    println("$cond \t nsamples = $nsamples")
    conditions = vcat(conditions,[cond for i in 1:nsamples])
    Xs = vcat(Xs, condXs .+ offs)
    Xg = vcat(Xg, condXg)
    Y = vcat(Y, condY) 
    offs += nsamples
    datasets = vcat(datasets, df)
end 
# remove extrema cells 
# get histogram of count per barcode

# keep most varying genes 20%

# library size normalization

# 
function normalise()
end

size(Xs)
X = (gpu(Int64.(Xs)), gpu(Int64.(Xg)))
Y = log10.(gpu(Float32.(Y)))
nsamples = maximum(X[1])
ngenes = maximum(X[2])
niter = 300_000
FE_params = generate_params_FE(nsamples, ngenes;emb_size=2, batchsize = 40_000, nsteps_dim_redux=niter)
model = FE_model(FE_params)
train_dev!(FE_params, X, Y, model, verbose = 1)

large_embed = cpu(model[1][1].weight)
large_labs =conditions 
fig = Figure(size = (1024,800));
ax = Axis(fig[1,1],title="Trained Embryoid Body Single-Cell RNA-seq data with FE during 20,000 steps", xlabel = "Cell-Embed-1", ylabel="Cell-Embed-2", aspect = 1);
colors = range(colorant"blue",stop=colorant"yellow", length=5)
# first plot train embed with circles.
for (i, group_lab) in enumerate(unique(large_labs))
    group = large_labs .== group_lab
    scatter!(ax, large_embed[1,group], large_embed[2,group], strokewidth = 0.1, color = colors[i], label = group_lab, marker = :circle)
end 
fig
CairoMakie.save("../figures/scRNAseq_embryoid_$niter.png", fig)
module FactorizedEmbeddings

using Flux
using Random
using ProgressMeter
using Statistics 
using Dates

export generate_params, fit, fit_transform, infer


function prep_FE(data::Matrix, device=gpu; order = "shuffled", verbose = 0)
    verbose > 0 && (println("Processing data..."))
    n, m = size(data)
    
    values = Array{Float32,2}(undef, (1, n * m))
    sample_index = Array{Int64,1}(undef, max(n * m, 1))
    gene_index = Array{Int64,1}(undef, max(n * m, 1))
    verbose > 0 && (p = Progress(n; showspeed=true))
    for i in 1:n
        for j in 1:m
            index = (i - 1) * m + j 
            values[1,index] = data[i,j] #
            sample_index[index] = i # Int
            gene_index[index] = j # Int 
            
        end
        verbose > 0 && next!(p)
    end 
    id_range = 1:length(values)
    order == "shuffled" ? id_range = Random.shuffle(id_range) : nothing
    return (device(sample_index[id_range]), device(gene_index[id_range])), device(vec(values[id_range]))
end 


function FE_model(params::Dict)
    emb_size_1 = params["emb_size_1"]
    emb_size_2 = params["emb_size_2"]
    a = emb_size_1 + emb_size_2 
    # b, c = params["fe_hl1_size"], params["fe_hl2_size"]#, params["fe_hl3_size"] ,params["fe_hl4_size"] ,params["fe_hl5_size"] 
    emb_layer_1 = gpu(Flux.Embedding(params["nsamples"], emb_size_1))
    emb_layer_2 = gpu(Flux.Embedding(params["ngenes"], emb_size_2))
    hlayers = []
    for (i,layer_size) in enumerate(params["fe_layers_size"][1:end])
        i == 1 ? inpsize = a : inpsize = params["fe_layers_size"][i - 1]
        push!(hlayers, Flux.Dense(inpsize, layer_size, relu))
    end 
    # hl1 = gpu(Flux.Dense(a, b, relu))
    # hl2 = gpu(Flux.Dense(b, c, relu))
    # hl3 = gpu(Flux.Dense(c, d, relu))
    # hl4 = gpu(Flux.Dense(d, e, relu))
    # hl5 = gpu(Flux.Dense(e, f, relu))
    outpl = gpu(Flux.Dense(params["fe_layers_size"][end], 1, identity))
    net = gpu(Flux.Chain(
        Flux.Parallel(vcat, emb_layer_1, emb_layer_2),
        hlayers..., outpl,
        vec))
    net 
end 

function my_cor(X::AbstractVector, Y::AbstractVector)
    sigma_X = std(X)
    sigma_Y = std(Y)
    mean_X = mean(X)
    mean_Y = mean(Y)
    cov = sum((X .- mean_X) .* (Y .- mean_Y)) / length(X)
    return cov / sigma_X / sigma_Y
end 

function reset_embedding_layer(FE_net, new_embed)
    hlayers = deepcopy(FE_net[2:end])
    test_FE = Flux.Chain(
        Flux.Parallel(vcat, 
            Flux.Embedding(new_embed),
            deepcopy(FE_net[1][2])
        ),
        hlayers...
    ) |> gpu
    return test_FE    
end 

function train!(params, X, Y, model;verbose = 0)
    verbose > 0 && (println("Training model..."))
    nsamples_batchsize = params["nsamples_batchsize"]
    batchsize = params["ngenes"] * nsamples_batchsize
    nminibatches = Int(floor(params["nsamples"] / nsamples_batchsize))
    opt = Flux.Adam(params["lr"])
    p = Progress(params["nsteps"]; showspeed=true)
    for iter in 1:params["nsteps"]
        # Stochastic gradient descent with minibatches
        cursor = (iter -1)  % nminibatches + 1
        
        batch_range = (cursor -1) * batchsize + 1 : cursor * batchsize
        X_, Y_ = (X[1][batch_range],X[2][batch_range]), Y[batch_range] # Access via "view" : quick
        ps = Flux.params(model)
       
        gs = gradient(ps) do 
            Flux.mse(model(X_), Y_) + params["l2"] * sum(p -> sum(abs2, p), ps) ## loss
        end
        lossval = Flux.mse(model(X_), Y_) + params["l2"] * sum(p -> sum(abs2, p), ps)
        pearson = my_cor(model(X_), Y_)
        Flux.update!(opt,ps, gs)
        # println("FE $(iter) epoch $(Int(ceil(iter / nminibatches))) - $cursor /$nminibatches - TRAIN loss: $(lossval)\tpearson r: $pearson ELAPSED: $((now() - start_timer).value / 1000 )") : nothing         
        next!(p; showvalues=[(:loss, lossval), (:pearson, pearson)])
    end
    return model 
end 


"""
    generate_params(X_data::AbstractArray; 
                    emb_size::Int=2, emb_size_2::Int=100, 
                    nsteps_dim_redux::Int=1000, l2_val::Float64=1e-7, 
                    fe_layers_size = [100, 50, 50]
               )

Function that takes input hyper-parameters and outputs a dictonary. 
"""
generate_params(X_data::AbstractArray; 
                    emb_size::Int=2, emb_size_2::Int=100, 
                    nsteps_dim_redux::Int=1000, l2_val::Float64=1e-7,
                    nsamples_batchsize::Int=1, 
                    fe_layers_size = [100, 50, 50]
               ) = return Dict( 
    ## run infos 
    # "session_id" => session_id,  "modelid" =>  "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
    # "outpath"=>outpath, 
    "machine_id"=>strip(read(`hostname`, String)), # "device" => "$(CUDA.device())", 
    ## data infos 
    "nsamples" =>size(X_data)[1], "ngenes"=> size(X_data)[2],  
    ## optim infos 
    "lr" => 5e-3, "l2" =>l2_val,"nsteps" => nsteps_dim_redux, "nsteps_inference" => Int(floor(nsteps_dim_redux * 0.1)), "nsamples_batchsize" => nsamples_batchsize,
    ## model infos
    "emb_size_1" => emb_size, "emb_size_2" => emb_size_2, "fe_layers_size"=> fe_layers_size,
    )



# fit function 
"""
    fit(X_data; dim_redux_size::Int=2, nsteps::Int=1000, l2::Float64=1e-7)

This function instanciates a Factorized Embeddings model with default or imputed parameters. Then trains the model on the input data and returns the trained model.
"""
function fit(X_data; dim_redux_size::Int=2, nsteps::Int=1000, l2::Float64=1e-7, verbose::Int=0)
    FE_params_dict = generate_params(X_data; 
        emb_size=dim_redux_size, 
        nsteps_dim_redux=nsteps, 
        l2_val=l2)
    X, Y = prep_FE(X_data, verbose=verbose);
    ## init model
    model = FE_model(FE_params_dict);
    # train loop
    model = train!(FE_params_dict, X, Y, model, verbose = verbose)
    return model 
end 

# fit function 
"""
    fit(X_data, FE_params::Dict)

This function instanciates a Factorized Embeddings model imputed hyper-parameter dictionary. Then trains the model on the input data and returns the trained model.
"""
function fit(X_data, FE_params::Dict;verbose::Int=0)
    
    X, Y = prep_FE(X_data;verbose = verbose);
    ## init model
    model = FE_model(FE_params);
    # train loop
    model = train!(FE_params, X, Y, model;verbose = verbose)
    return model 
end 


# fit_transform function 
"""
    fit_transform(X_data; dim_redux_size::Int=2, nsteps::Int=1000, l2::Float64=1e-7, verbose::Int = 0)

This function instanciates a Factorized Embeddings model with default or imputed parameters. Then trains the model on the input data and returns the dimensionality-reduced sample embedding.
"""
function fit_transform(X_data; dim_redux_size::Int=2, nsteps::Int=1000, l2::Float64=1e-7, verbose::Int = 0)
    model = fit(X_data, dim_redux_size=dim_redux_size, nsteps = nsteps, l2=l2, verbose = verbose)
    return cpu(model[1][1].weight) 
end 

# fit_transform function 
"""
    fit_transform(X_data, FE_params::Dict;verbose::Int=0)

This function instanciates a Factorized Embeddings model imputed hyper-parameter dictionary. Then trains the model on the input data and returns the dimensionality-reduced sample embedding.
"""
function fit_transform(X_data, FE_params::Dict;verbose::Int=0)
    model = fit(X_data, FE_params, verbose = verbose)
    return cpu(model[1][1].weight) 
end 

"""
   infer(trained_FE, train_data, test_data, params_dict;verbose=0)

Infers new data with the pre-trained model. Input parameters.
trained_FE: the pre-trained Flux DNN model.
train_data: the training dataset.
test_data: the test dataset. 
params_dict: Dictionary of hyper-parameters that was set during the training phase.
"""
function infer(trained_FE, train_data, test_data, params_dict::Dict;verbose=0)
    start_timer = now()
    tst_elapsed = []
    ## generate X and Y test data. 
    X_train, Y_train = prep_FE(train_data;order="per_sample", verbose = verbose)
    nsamples_batchsize = 1
    batchsize = params_dict["ngenes"] * nsamples_batchsize
    nminibatches = Int(floor(params_dict["nsamples"] / nsamples_batchsize))
    MM = zeros((size(train_data)))
    verbose > 0 && (println("Preparing infered train profiles matrix..."))
    for iter in 1:nminibatches
        batch_range = (iter-1) * batchsize + 1 : iter * batchsize
        X_, Y_ = (X_train[1][batch_range],X_train[2][batch_range]), Y_train[batch_range]
        MM[iter, :] .= vec(cpu(trained_FE(X_)))
    end 
    infered_train = gpu(MM)
    push!(tst_elapsed, (now() - start_timer).value / 1000 )
    verbose > 0 && (println("Infered train profiles matrix. $(tst_elapsed[end]) s\nComputing distances to target samples..."))
    new_embed = zeros(params_dict["emb_size_1"], size(test_data)[1])
    test_data_G = gpu(test_data)
    p = Progress(size(test_data)[1]; showspeed=true)
    for infer_patient_id in 1:size(test_data)[1]
        EuclideanDists = sum((infered_train .- vec(test_data_G[infer_patient_id, :])') .^ 2, dims = 2)
        new_embed[:,infer_patient_id] .= cpu(trained_FE[1][1].weight[:,findfirst(EuclideanDists .== minimum(EuclideanDists))[1]])
        next!(p; showvalues=[(:elapsed,(now() - start_timer).value / 1000)])
    end 
    push!(tst_elapsed, (now() - start_timer).value / 1000 )
    verbose > 0 && (println("distances to target sample. $(tst_elapsed[end]) s\nOptimizing model with test samples optimal initial positions..."))

    inference_model = reset_embedding_layer(trained_FE, new_embed)
    model_phase_1 = cpu(inference_model[1][1].weight)
    # fig1 = plot_train_test_patient_embed(trained_FE, inference_model, labs, train_ids, test_ids, params_dict);
    X_test, Y_test = prep_FE(test_data;order = "per_sample", verbose = verbose);
    nsamples_batchsize = params_dict["nsamples_batchsize"]
    batchsize = params_dict["ngenes"] * nsamples_batchsize
    nminibatches = Int(floor(length(Y_test) / batchsize))
    
    opt = Flux.Adam(params_dict["lr"])
    p = Progress(params_dict["nsteps_inference"]; showspeed=true)
    for iter in 1:params_dict["nsteps_inference"]
        cursor = (iter -1)  % nminibatches + 1
        mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, length(Y_test)))
        X_, Y_ = (X_test[1][mb_ids],X_test[2][mb_ids]), Y_test[mb_ids]
        
        ps = Flux.params(inference_model[1][1])
        gs = gradient(ps) do 
            Flux.mse(inference_model(X_), Y_) + params_dict["l2"] * sum(p -> sum(abs2, p), ps)
        end
        lossval = Flux.mse(inference_model(X_), Y_) + params_dict["l2"] * sum(p -> sum(abs2, p), ps)
        pearson = my_cor(inference_model(X_), Y_)
        Flux.update!(opt,ps, gs)
        push!(tst_elapsed, (now() - start_timer).value / 1000 )
        # iter % 1000 == 0 ?  println("$(iter) epoch $(Int(ceil(iter / nminibatches))) - $cursor /$nminibatches - TRAIN loss: $(lossval)\tpearson r: $pearson \t elapsed: $(tst_elapsed[end]) s") : nothing
        next!(p; showvalues=[(:step,iter), (:elapsed,(now() - start_timer).value / 1000), (:loss,lossval),(:pearson, pearson)])
    end 
    verbose > 0 && (println("Final embedding $(tst_elapsed[end]) s"))
    return inference_model, model_phase_1
end 

end
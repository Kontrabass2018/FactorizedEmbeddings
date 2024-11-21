module FactorizedEmbeddings

# Write your package code here.

export fit, fit_transform

function prep_FE(data::Matrix, device=gpu; order = "shuffled")
    n, m = size(data)
    
    values = Array{Float32,2}(undef, (1, n * m))
    sample_index = Array{Int64,1}(undef, max(n * m, 1))
    gene_index = Array{Int64,1}(undef, max(n * m, 1))
    
    for i in 1:n
        for j in 1:m
            index = (i - 1) * m + j 
            values[1,index] = data[i,j] #
            sample_index[index] = i # Int
            gene_index[index] = j # Int 
            
        end
    end 
    id_range = 1:length(values)
    order == "shuffled" ? id_range = shuffle(id_range) : nothing
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


function train!(params, X, Y, model)
    nsamples_batchsize = params["nsamples_batchsize"]
    batchsize = params["ngenes"] * nsamples_batchsize
    nminibatches = Int(floor(params["nsamples"] / nsamples_batchsize))
    opt = Flux.ADAM(params["lr"])
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

generate_params(X_data, emb_size, nsteps_dim_redux, l2_val) = return Dict( 
    ## run infos 
    # "session_id" => session_id,  "modelid" =>  "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
    # "outpath"=>outpath, 
    "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())", 
    ## data infos 
    "nsamples" =>size(X_data)[1], "ngenes"=> size(X_data)[2],  
    ## optim infos 
    "lr" => 5e-3, "l2" =>l2_val,"nsteps" => nsteps_dim_redux, "nsteps_inference" => Int(floor(nsteps_dim_redux * 0.1)), "nsamples_batchsize" => 4,
    ## model infos
    "emb_size_1" => emb_size, "emb_size_2" => 100, "fe_layers_size"=> [100, 50, 50],
    )

# fit function 
"""
    fit(X_data; dim_redux_size::Int=2, nsteps::Int=1000, l2::Float64=1e-7)

This function instanciates a Factorized Embeddings model with default or imputed parameters. Then trains the model on the input data and returns the trained model.
"""
function fit(X_data; dim_redux_size::Int=2, nsteps::Int=1000, l2::Float64=1e-7)
    FE_params_dict = generate_params(X_data, dim_redux_size, nsteps, l2)
    X, Y = prep_FE(X_data);
    ## init model
    model = FE_model(FE_params_dict);
    # train loop
    model = train!(FE_params_dict, X, Y, model)
    return model 
end 

# fit_transform function 
"""
    fit_transform(X_data; dim_redux_size::Int=2, nsteps::Int=1000, l2::Float64=1e-7)

This function instanciates a Factorized Embeddings model with default or imputed parameters. Then trains the model on the input data and returns the dimensionality-reduced sample embedding.
"""
function fit_transform(X_data; dim_redux_size::Int=2, nsteps::Int=1000, l2::Float64=1e-7)
    model = fit(X_data, dim_redux_size=dim_redux_size, nsteps = nsteps, l2=l2)
    return cpu(model[1][1].weight) 
end 

end
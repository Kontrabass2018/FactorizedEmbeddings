using Dates
using ProgressMeter

struct FE_model
    embed
    dnn 
end

struct EmbedL
    emb1
    emb2
end 


function (m::EmbedL)(X)
    return vcat(m.emb1(X[1]), m.emb2(X[2])) 
end 

function (m::FE_model)(X)
    return vec(m.dnn(m.embed(X)))
end 

Flux.@layer EmbedL trainable=(emb1,)
Flux.@layer FE_model trainable=(embed)


function init_inference_FE(trained_FE, new_embed)
    DNN = deepcopy(trained_FE[2:end])
    emb1 = Flux.Embedding(new_embed) |> gpu
    emb2 = deepcopy(trained_FE[1][2]) |> gpu
    EMBL = EmbedL(emb1, emb2)
    inference_model = FE_model(EMBL, DNN)
    return inference_model
end 

function my_cor(X::AbstractVector, Y::AbstractVector)
    sigma_X = std(X)
    sigma_Y = std(Y)
    mean_X = mean(X)
    mean_Y = mean(Y)
    cov = sum((X .- mean_X) .* (Y .- mean_Y)) / length(X)
    return cov / sigma_X / sigma_Y
end 

function infer_dev(trained_FE, train_data, test_data, params_dict::Dict;verbose=0)
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
    # new_embed
    # inference_model = FactorizedEmbeddings.reset_embedding_layer(trained_FE, new_embed)
    # model_phase_1 = cpu(inference_model[1][1].weight)
    # # fig1 = plot_train_test_patient_embed(trained_FE, inference_model, labs, train_ids, test_ids, params_dict);
    # X_test, Y_test = prep_FE(test_data;order = "per_sample", verbose = verbose);
    # nsamples_batchsize = params_dict["nsamples_batchsize"]
    # batchsize = params_dict["ngenes"] * nsamples_batchsize
    # nminibatches = Int(floor(length(Y_test) / batchsize))
    
    # opt = Flux.Adam(params_dict["lr"])
    # state = Flux.setup(opt, inference_model) |> gpu 
    # p = Progress(params_dict["nsteps_inference"]; showspeed=true)
    # for iter in 1:params_dict["nsteps_inference"]
    #     cursor = (iter -1)  % nminibatches + 1
    #     mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, length(Y_test)))
    #     X_, Y_ = (X_test[1][mb_ids],X_test[2][mb_ids]), Y_test[mb_ids]
        
    #     grads = Flux.gradient(inference_model) do m 
    #         Flux.mse(m(X_), Y_) + params_dict["l2"] * sum(sum.(abs2, Flux.trainables(m))) 
    #     end
    #     lossval = Flux.mse(inference_model(X_), Y_) + params_dict["l2"] * sum(sum.(abs2, Flux.trainables(inference_model))) 
    #     pearson = my_cor(inference_model(X_), Y_)
    #     Flux.update!(state, inference_model, grads[1])
    #     push!(tst_elapsed, (now() - start_timer).value / 1000 )
    #     # iter % 1000 == 0 ?  println("$(iter) epoch $(Int(ceil(iter / nminibatches))) - $cursor /$nminibatches - TRAIN loss: $(lossval)\tpearson r: $pearson \t elapsed: $(tst_elapsed[end]) s") : nothing
    #     next!(p; showvalues=[(:step,iter), (:elapsed,(now() - start_timer).value / 1000), (:loss,lossval),(:pearson, pearson)])
    # end 
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
    # verbose > 0 && (println("Final embedding $(tst_elapsed[end]) s"))
    return new_embed
end 
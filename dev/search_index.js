var documenterSearchIndex = {"docs":
[{"location":"#Factorized-Embeddings-method","page":"-","title":"Factorized Embeddings method","text":"","category":"section"},{"location":"#Main-methods","page":"-","title":"Main methods","text":"","category":"section"},{"location":"","page":"-","title":"-","text":"generate_params","category":"page"},{"location":"#FactorizedEmbeddings.generate_params","page":"-","title":"FactorizedEmbeddings.generate_params","text":"generate_params(X_data::AbstractArray; \n                emb_size::Int, emb_size_2::Int = 100, \n                nsteps_dim_redux::Int, l2_val::Float64, \n                fe_layers_size = [100, 50, 50]\n           )\n\nFunction that takes input hyper-parameters and outputs a dictonary. \n\n\n\n\n\n","category":"function"},{"location":"","page":"-","title":"-","text":"fit","category":"page"},{"location":"#FactorizedEmbeddings.fit","page":"-","title":"FactorizedEmbeddings.fit","text":"fit(X_data; dim_redux_size::Int=2, nsteps::Int=1000, l2::Float64=1e-7)\n\nThis function instanciates a Factorized Embeddings model with default or imputed parameters. Then trains the model on the input data and returns the trained model.\n\n\n\n\n\nfit(X_data, FE_params::Dict)\n\nThis function instanciates a Factorized Embeddings model imputed hyper-parameter dictionary. Then trains the model on the input data and returns the trained model.\n\n\n\n\n\n","category":"function"},{"location":"","page":"-","title":"-","text":"fit_transform","category":"page"},{"location":"#FactorizedEmbeddings.fit_transform","page":"-","title":"FactorizedEmbeddings.fit_transform","text":"fit_transform(X_data; FE_params::Dict)\n\nThis function instanciates a Factorized Embeddings model imputed hyper-parameter dictionary. Then trains the model on the input data and returns the dimensionality-reduced sample embedding.\n\n\n\n\n\n","category":"function"},{"location":"","page":"-","title":"-","text":"infer","category":"page"},{"location":"#FactorizedEmbeddings.infer","page":"-","title":"FactorizedEmbeddings.infer","text":"infer(trainedFE, traindata, trainids, testdata, testids,  samples, genes, paramsdict)\n\nInfers new data with the pre-trained model.\n\n\n\n\n\n","category":"function"}]
}

var documenterSearchIndex = {"docs":
[{"location":"guide/#Factorized-Embeddings-method","page":"Guide","title":"Factorized Embeddings method","text":"","category":"section"},{"location":"guide/#Package-Guide","page":"Guide","title":"Package Guide","text":"","category":"section"},{"location":"guide/#Installation","page":"Guide","title":"Installation","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"At this point, this package is still in its development stage, so the package is not yet available through the general Julia registry. Users can use the development github repository link to use the latest version of the method.","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"pkg>add https://github.com/Kontrabass2018/FactorizedEmbeddings","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Once the package is downloaded and added to your working environment, standard package importation can be called.","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"using FactorizedEmbeddings","category":"page"},{"location":"guide/#Example-of-usage","page":"Guide","title":"Example of usage","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"train_redux = fit_transform(train_data, verbose = 1);","category":"page"},{"location":"guide/#Advanced-usage","page":"Guide","title":"Advanced usage","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"model_params = generate_params(train_data, nsteps_dim_redux = 20_000)\nmodel = FE_model(model_params);\nfit!(train_data, model, model_params)\ninfer_model = infer(model, train_data, test_data, model_params, verbose = 1)","category":"page"},{"location":"#Factorized-Embeddings-method","page":"Home","title":"Factorized Embeddings method","text":"","category":"section"},{"location":"#Description","page":"Home","title":"Description","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Factorized Embeddings is described in this article by Trofimov et al. 2020. This model is a self-supervised deep neural network that uses tensor factorization to simultaneously learn gene and sample representation spaces. This type of network is tailored to work with large RNA sequencing data, but can be applied to any type of large multivariate data. The FE model treats both factors (typically samples and gene) as factors contributing to characterizing the values (ie. gene expression data). This architecture generates sample representations that can be used for auxiliary tasks such as visualization or classification.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The code in this package is based on the first Factorized Embeddings analysis scripts according to the code provided by Trofimov et al. (2020) GitHub link. The demonstrated functionalities of Factorized Embeddings are implemented in the Julia programming language julialang.org using the Flux library Innes 2018. The main features of the code include: •\tGeneration of low-dimensional sample embeddings that can be used for auxiliary tasks like 2D visualization or classification. •\tHighly configurable models via a dictionary of hyperparameters provided by the user. •\tDuring network optimization, models, learning curves, reconstruction performance, and, if possible, 2D sample embedding visualizations are recorded. •\tOnce trained, the model can infer new points. •\tA functionality is also available for imputing the sample embedding space in 2D.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Model optimizations were tested using SGD with the ADAM optimizer (Kingma and Ba 2017) on GPUs (V100-SMX2, 32 GB) on servers with 64 GB RAM. It is important to note that this implementation requires access to a GPU for optimal performance. The code can be adapted for specific server constraints, such as reducing mini-batch sizes and limiting the number of network parameters. Details about the required package installations for GPU usage are also provided in this guide.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Schematic of Factorized Embeddings","category":"page"},{"location":"#Public-Interface","page":"Home","title":"Public Interface","text":"","category":"section"},{"location":"#Main-methods","page":"Home","title":"Main methods","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"generate_params","category":"page"},{"location":"#FactorizedEmbeddings.generate_params","page":"Home","title":"FactorizedEmbeddings.generate_params","text":"generate_params(X_data::AbstractArray; \n                emb_size::Int=2, emb_size_2::Int=100, \n                nsteps_dim_redux::Int=1000, l2_val::Float64=1e-7, \n                fe_layers_size = [100, 50, 50]\n           )\n\nFunction that takes input hyper-parameters and outputs a dictonary. \n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"fit","category":"page"},{"location":"#FactorizedEmbeddings.fit","page":"Home","title":"FactorizedEmbeddings.fit","text":"fit(X_data; dim_redux_size::Int=2, nsteps::Int=1000, l2::Float64=1e-7)\n\nThis function instanciates a Factorized Embeddings model with default or imputed parameters. Then trains the model on the input data and returns the trained model.\n\n\n\n\n\nfit(X_data, FE_params::Dict)\n\nThis function instanciates a Factorized Embeddings model imputed hyper-parameter dictionary. Then trains the model on the input data and returns the trained model.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"fit_transform","category":"page"},{"location":"#FactorizedEmbeddings.fit_transform","page":"Home","title":"FactorizedEmbeddings.fit_transform","text":"fit_transform(X_data; dim_redux_size::Int=2, nsteps::Int=1000, l2::Float64=1e-7, verbose::Int = 0)\n\nThis function instanciates a Factorized Embeddings model with default or imputed parameters. Then trains the model on the input data and returns the dimensionality-reduced sample embedding.\n\n\n\n\n\nfit_transform(X_data, FE_params::Dict;verbose::Int=0)\n\nThis function instanciates a Factorized Embeddings model imputed hyper-parameter dictionary. Then trains the model on the input data and returns the dimensionality-reduced sample embedding.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"infer","category":"page"},{"location":"#FactorizedEmbeddings.infer","page":"Home","title":"FactorizedEmbeddings.infer","text":"infer(trainedFE, traindata, testdata, paramsdict;verbose=0)\n\nInfers new data with the pre-trained model. Input parameters. trainedFE: the pre-trained Flux DNN model. traindata: the training dataset. testdata: the test dataset.  paramsdict: Dictionary of hyper-parameters that was set during the training phase.\n\n\n\n\n\n","category":"function"},{"location":"#Other-utility-methods","page":"Home","title":"Other utility methods","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"fit!","category":"page"},{"location":"#FactorizedEmbeddings.fit!","page":"Home","title":"FactorizedEmbeddings.fit!","text":"fit!(X_data, model, FE_params::Dict)\n\nThis function uses the inputed Factorized Embeddings model with pre-defined and imputed hyper-parameter dictionary. Then trains the model on the input data and returns the trained model.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"reset_embedding_layer","category":"page"},{"location":"#FactorizedEmbeddings.reset_embedding_layer","page":"Home","title":"FactorizedEmbeddings.reset_embedding_layer","text":"function reset_embedding_layer!(FE_net, new_embed; cp_dev=gpu)\n\nThis methods copies the imputed model and returns a model with inputed pre-defined embedding layer.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"This function allows to initialise the sample embedding with a certain pre-defined embedding.","category":"page"},{"location":"","page":"Home","title":"Home","text":"train!","category":"page"},{"location":"#FactorizedEmbeddings.train!","page":"Home","title":"FactorizedEmbeddings.train!","text":"function train!(params, X, Y, model;verbose = 0)\n\nThis methods trains a model with the X and Y training data and returns the trained model.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"FE_model","category":"page"},{"location":"#FactorizedEmbeddings.FE_model","page":"Home","title":"FactorizedEmbeddings.FE_model","text":"FE_model(params::Dict)\n\nThis method takes as input a dictionary of hyperparameters instantiates and returns a Flux.Chain type DNN model.  This model can then be fed to the train! method to fit the model's parameters to the data. \n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"prep_FE","category":"page"},{"location":"#FactorizedEmbeddings.prep_FE","page":"Home","title":"FactorizedEmbeddings.prep_FE","text":"prep_FE(data::Matrix, device=gpu; order = \"shuffled\", verbose = 0)\n\nThis method takes the data matrix as input and outputs the vectors X containing two vectors of samples indices and gene indices and Y containing the expression values of the sample and the gene.\n\n\n\n\n\n","category":"function"}]
}

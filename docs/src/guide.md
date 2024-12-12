# Factorized Embeddings method
## Package Guide
### Installation
At this point, this package is still in its development stage, so the package is not yet available through the general Julia registry. Users can use the development github repository link to use the latest version of the method.
```julia
pkg>add https://github.com/Kontrabass2018/FactorizedEmbeddings
```

Once the package is downloaded and added to your working environment, standard package importation can be called.
```julia
using FactorizedEmbeddings
```

### Example of usage
```julia
train_redux = fit_transform(train_data, verbose = 1);
```

### Advanced usage

```julia
model_params = generate_params(train_data, nsteps_dim_redux = 20_000)
model = FE_model(model_params);
fit!(train_data, model, model_params)
infer_model = infer(model, train_data, test_data, model_params, verbose = 1)
```
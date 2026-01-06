# GPNSF
A model using GP priors and NMF to facilitate multi-omics data integration.
# Model features
* Reconstruct original count matrices using designed likelihood estimation. Modality 1 is set to be negative binomial distributed (for RNA), modality 2 is given three options for users to choose: negative binomial distributed, binomial distributed and Poisson distributed.
* A GP prior is added in the low-dimensional latent space for modeling spatial similarity. Sparse GP and variational inference are conducted to make the model scalable when dealing with large dataset (e.g. >5k spots).
The specific model description and detailed derivation of the loss function are included in [model_description.pdf](https://github.com/dd719/gpnsf/blob/main/model_description.pdf).
# Model architecture
![](model_architecture.png)

# Quick start
* import packages
```
from GPNAF.model import *
from GPNAF.utils import *
```
* prepare the data (load X_1, X_2 and S from your Anndata objects)
```
S_t = to_torch_tensor(S, device, dtype=torch.float32)
X1_t = to_torch_tensor(X_1, device, dtype=torch.float32)
X2_t = to_torch_tensor(X_2, device, dtype=torch.float32)
```
* training
```
model = GPNSFModel(S=S_t, p=p, q=q, K=K, M=M).to(device)
model = train_model(model=model, X1_t=X1_t, X2_t=X2_t)
```
* add latent representations as adata.obs
```
add_latent_to_adata(adata1, model) # adata1 is either your Anndata object
```
A complete demo is shown in [run_GPNSF/simulation_1.ipynb](https://github.com/dd719/gpnsf/blob/main/run_GPNSF/simulation_1.ipynb).


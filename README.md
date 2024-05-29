# QDGP_GI_ImageTasks
The repository is the implementation of QDGP for ghost imaging and typical image tasks. Partial of the code is borrowed from DGP implementation [https://github.com/ajbrock/BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch) and [https://github.com/XingangPan/deep-generative-prior](https://github.com/XingangPan/deep-generative-prior)

## Quantum Implementation
The inplementation of quantum neural networks is based on tensorcircuit, which supports hybrid quantum-classical quantum machine learning tasks. To obtain high performance of quantum machine learning, we recommend using tensorflow backend to construct the quantum circuits. The code can be executed on GPU devices and the maximal memory requirement is up to 12GB.

## MPS Circuit implementation
In case using MPS circuit as the backend, one should set the backend into MPSCircuit in the quantum circuit model. Besides, since MPS uses SVD operation, one may also set the dtype of the tensorflow or jax backend into complex128 to ensure the numerical stability. The bond dimension should not be too large to ensure a relative fast running.

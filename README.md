# QDGP_GI_ImageTasks
The repository is the implementation of QDGP for ghost imaging and typical image tasks. Partial of the code is borrowed from DGP implementation [https://github.com/ajbrock/BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch) and [https://github.com/XingangPan/deep-generative-prior](https://github.com/XingangPan/deep-generative-prior)

## Quantum Implementation
The inplementation of quantum neural networks is based on tensorcircuit, which supports hybrid quantum-classical quantum machine learning tasks. To obtain high performance of quantum machine learning, we recommend using tensorflow backend to construct the quantum circuits. The code can be executed on GPU devices and the maximal memory requirement is up to 12GB.

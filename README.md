# IS-FedVAE
Official code repository for the paper "IMPORTANCE SAMPLING BASED FEDERATED UNSUPERVISED REPRESENTATION LEARNING" accepted at IEEE ICASSP 2024, Seoul, South Korea.

## Setup the environment
Use the IS_FedVAE.yml file or run the following command :
 `pip install -r requirements.txt`

## Setup the dataset
An example is shown below. Running the following command creates 10 partitions CIFAR10 dataset with the Dirichlet parameter alpha=0.1.

`python sampler.py --dataset="CIFAR10" --n_clients=10 --alpha=0.1`

All the parameters can be changed directly in the IS-FedVAE.py file.



# Rapid-SPN
SumProduct Network under Random Projection

Prajay Shetty </br>
University of Georgia </br>
Department of Computer Science </br>

Contributions:</br>
Dr.Doshi provided the teaching for SPN </br>
Dr.Kristian provided the idea for using SPN with RP-Tree and taught SPN </br>
Prajay Shetty had the novelty for the approach of combining RP-Tree and SPN and also wrote down theory behind it and performed analysis and wrote code. </br>
Fabrzio Ventola, supported by running experiments in the European Cluster based on the input provided by Prajay Shetty. </br>

To be added in the final paper.</br>

For proper usage:
Clone spflow https://github.com/SPFlow/SPFlow and run src/create_pip_dist.sh and install pip file from dist directory.
Please install  scipy==1.11.4
To Execute:

1.For normal scripts

cd scripts

python3 cifar_10_spnrp.py no_of_rows no_of_columns

2. For PCA scripts
   
cd scripts

python3 cifar_pca.py no_of_rows no_of_columns no_of_components

Preprint Zenodo link https://zenodo.org/records/13726892

https://doi.org/10.5281/zenodo.13731764

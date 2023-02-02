# MHAGP
## Predicting disease genes based on multi-head attention fusion

This paper proposes an approach to predict the pathogenic gene based on multi-head attention fusion (MHAGP). Firstly, the heterogeneous biological information network of disease genes is constructed by integrating multiple biomedical knowledge databases. Secondly, two graph representation learning algorithms are used to capture the feature vectors of gene-disease pairs from the network, and the features are fused by introducing multi-head attention. Finally, we use a multi-layer perceptron model to predict the gene-disease association.

**Environment Requirement**

The code has been tested running under Python 3.7. The  required main packages are as follows:
* torch=1.11
* networkx=2.0
* node2vec=0.4.3
* gensim=3.0.1
* Scikit Learn
* numpy=1.19
* pandas>=1.0
* h5py=2.10
* openssl=1.1


**Python implementation files for MHAGP**
     1. line_embedding.py and node2vec_embedding.py are used to extract three network features.

     2. feature_concatenate.ipynb - Jupyter notebook for concatenate the gene-disease features of the three networks extracted by line and node2vec algorithm.

     3. attenetion_prediction.ipynb- Jupyter notebook for the gene-disease association prediction.

**Usage**

### Cloning the repo
Code tested only in NVIDIA GeForce RTX 3090.

### shell
git clone https://github.com/axing209/MHAGP.git

For any doubts or suggestions please contact：
axing209729@gmail.com
ldr@stu.xju.edu.cn


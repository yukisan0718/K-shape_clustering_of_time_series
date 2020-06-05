K-shape clustering of time series
====

## Overview
Implementation of clustering algorithms specialized in time series signals.

The "K-shape_clustering.py" is using the k-shape, which is a clustering algorithm relying on the shape-based distance (SBD) specialized in time series signals [1].

The "Fuzzy_c-shape_clustering.py" is using the fuzzy c-shape, which is an advanced version of the k-shape proposed in [2]. A point to note is that the convergence is not guaranteed in the algorithm.

Both can apply to any clustering application, especially on time series signals.


## Requirement
matplotlib 3.1.0

numpy 1.18.1

scipy 1.4.1


## Dataset preparation
You can apply these clustering methods to any application you want. An example of foreign exchange rate between USD and JPY has been prepared for the demonstration.


## References
[1] J. Paparrizos and L. Gravano: 'K-Shape: Efficient and Accurate Clustering of Time Series', In Proceedings of the ACM SIGMOD International Conference on Management of Data, pp.1855–1870, (2015)

[2] F. Fahiman, J. C. Bezdek, S. M. Erfani, M. Palaniswami, and C. Leckie: 'Fuzzy C-Shape: A New Algorithm for Clustering Finite Time Series Waveforms', In Proceedings of the IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), pp.1–8, (2017)
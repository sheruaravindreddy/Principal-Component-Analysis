# Principal-Component-Analysis
Transforming data into a new dimensional subspace and then using the new data for model building
Instead of building the model from scratch, sklearn library is employed in doing this
But this could be built easily from scratch as well:
1) Find the covariance matrix using numpy
2) Then standardize the data applying (X-Mu)/Sigma. This could be achieved easily using pandas.
3) Then using the lin_alg function find the eigen values and eigen vectors. Sort the eigen values in descending order and pick the highest releavant vectors.
4) Transpose them to obtain the newly created transformed non-collinear features.

# Unsupervised learning with Self Organizing Maps (SOM)
# Dataset 2: Wine.txt



# Import modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import somoclu



# Load dataset

# Load training data
filename = 'input/Wine.txt'
df = pd.read_csv(filename, delimiter=',', header=0)

features = ['Alcohol','Malic_Acid','Ash','Ash_Alcanity','Magnesium','Total_Phenols','Flavanoids','Nonflavanoid_Phenols','Proanthocyanins','Color_Intensity','Hue','OD280','Proline']
target = ['Customer_Segment']

# Separating out the features
x = df.loc[:, features].values

# Separating out the target (class)
y = df.loc[:, target].values

print(x)



# Planar maps
# Example: https://somoclu.readthedocs.io/en/stable/example.html#toroid-topology-hexagonal-grid
print('Planar maps...')

# We train Somoclu with default parameter settings, asking for a large
# map that qualifies as an emergent self-organizing map for this data:
n_rows, n_columns = 200, 200
som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False)
som.train(x)

# We plot the component planes of the trained codebook of the ESOM:
som.view_component_planes()

# We can plot the U-Matrix, together with the best matching units
# for each data point. We color code the units with the classes of
# the data points and also add the labels of the data points.
colors = ['red','green','blue','purple','yellow','black','cyan']
som.view_umatrix(bestmatches=True, labels=y, filename='output_som_d2/planar_map__umatrix')

# We can also zoom into a region of interest, for instance,
# the dense lower right corner:
som.view_umatrix(bestmatches=True, filename='output_som_d2/planar_map__umatrix_zoom', labels=y,
                 zoom=((50, n_rows), (100, n_columns)))




# Toroid topology, hexagonal grid
# Example: https://somoclu.readthedocs.io/en/stable/example.html#toroid-topology-hexagonal-grid
print('Toroid topology, hexagonal grid...')

# We can repeat the above with a toroid topology by specifying the map
# type as follows:
som = somoclu.Somoclu(n_columns, n_rows, maptype="toroid",
                      compactsupport=False)
som.train(x)
som.view_umatrix(bestmatches=True, filename='output_som_d2/toroid_map__umatrix')



# Initialization with principal component analysis and clustering the results
# Example: https://somoclu.readthedocs.io/en/stable/example.html#initialization-with-principal-component-analysis-and-clustering-the-results
print('Initialization with principal component analysis and clustering the results...')

# We can pass an initial codebook of our choice,but we can also ask Somoclu
# to initialize the codebook with vectors from the subspace spanned by the
# first two eigenvalues of the correlation matrix. To do this, we need to pass
# an optional argument to the constructor:
som = somoclu.Somoclu(n_columns, n_rows, maptype="toroid",
                      compactsupport=False, initialization="pca")
som.train(x)
som.view_umatrix(bestmatches=True, filename='output_som_d2/toroid_map_pca__umatrix')

# While one would expect entirely deterministic results on repeated runs
#  with the initialization based on PCA, this is not the case. The order
#  in which the data instances arrive matters: since Somoclu uses multiple
#  cores, there is no control over the order of each batch, hence the maps
#  will show small variation even with a PCA initalization.

# We can also postprocess the codebook with an arbitrary clustering algorithm
#  that is included in scikit-learn. The default algorithm is K-means with eight
#  clusters. After clustering, the labels for each node are available in the
#  SOM object in the clusters class variable. If we do not pass colors to the
#  matrix viewing functions and clustering is already done, the plotting
#  routines automatically color the best matching units according to the
#  clustering structure.

som.cluster()
som.view_umatrix(bestmatches=True)
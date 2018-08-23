import random
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

a = [random.randint(1,40) for i in range(20)]
b = [random.randint(100,4000) for i in range(20)]
c = [(1.2*a[i] + 1.36*b[i]) for i in range(20)]

df = pd.DataFrame()

count = 1
for i in [a,b,c]:
    df["col_%d"%(count)] = pd.Series(i)
    count += 1
print df


X = np.mat(df)

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
pca.fit(X_std)

print [float(i) for i in (pca.explained_variance_ratio_)*100]


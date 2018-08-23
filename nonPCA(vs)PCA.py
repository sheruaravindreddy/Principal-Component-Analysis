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

#preparing X
X = np.mat(df)

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(X_std)

#preparing transformed X
pca_X = (pca.transform(X)) 

#preparing dependent variable Y
rand_list = [random.uniform(0,1) for i in range(20)]
rand_list = np.multiply(np.array(c),np.array(rand_list))
Y = []
for i in rand_list:
    if i < np.mean(rand_list):
        Y.append(0)
    else:
        Y.append(1)
        

# fitting both the idv's to LR
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import accuracy_score
log = LogisticRegression()

#using simple IDV's
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)
log.fit(X_train,Y_train)
accuracy_nonpca = accuracy_score(Y_test,log.predict(X_test))
print accuracy_nonpca

#using pca transformed IDV's
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(pca_X, Y, test_size=0.2)
log.fit(X_train,Y_train)
accuracy_pca = accuracy_score(Y_test,log.predict(X_test))
print accuracy_pca
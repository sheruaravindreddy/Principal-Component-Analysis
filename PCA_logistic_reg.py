import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, model_selection
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

df_input = pd.read_csv('lg_data.csv')
##df_v5 = df_input[pd.notnull(df_input['v5_nps'])]

df = df_input.drop(columns = ['Unnamed: 0', 'LoanName', 'loan__Due_Date__c', 'loan__Frequency_of_Loan_Payment__c',
                   'loan__Payment_Date__c', 'loan__Payment_Satisfied__c',
                   'loan__Disbursal_Date__c','observation_date', 'observation_interval',
                   'loan__Loan_Amount__c','loan__Closed_Date__c', 'loan__Charged_Off_Date__c'])

df = df.drop(columns = ['application_started', 'application_submitted',
                        'DL_verification_started', 'DL_verification_submitted',
                        'email_verification_started', 'email_verification_submitted',
                        'loandesign_started', 'loandesign_submitted', 'esign_started',
                        'esign_submitted', 'application_difference',
                        'DL_verification_difference', 'email_difference',
                        'loandesign_difference', 'esign_difference', 'v5_nps'])

df = df.dropna()
df = df.drop(df[df.dv == -99].index)
df = df.reset_index()
df = df.drop(columns = ['index'])

##...Selecting the dependent feature(variable) as Y and remaining features as X
X = df.drop(columns = ['dv'])
Y = df['dv']

from sklearn.preprocessing import LabelEncoder
le = preprocessing.LabelEncoder()
X = X.apply(LabelEncoder().fit_transform)


from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X.T)


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(X_std)
X_DimRed = pca.transform(X_std)

X_orig_dim = (np.dot(X,X_DimRed))

##print(pca.mean_)
##print(pca.explained_variance_)
##print (pca.singular_values_)
##print(pca.n_components_)
##print(pca.noise_variance_)
print(pca.components_)

print sum(pca.explained_variance_ratio_)

X = X_orig_dim





##...Splitting the dataset into training and testing samples
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)


##...Preparing the data using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler()
X_train_minmax = min_max.fit_transform(X_train)
X_test_minmax = min_max.fit_transform(X_test)

##...Preparing the data using Normalization
from sklearn.preprocessing import scale
X_train_scale = scale(X_train)
X_test_scale = scale(X_test)


log = LogisticRegression(penalty='l2', C=.01, dual=False, tol=0.0001, fit_intercept=True,
                         intercept_scaling=1, class_weight=None, random_state=None,
                         solver='liblinear', max_iter=100, multi_class='ovr', verbose=0,
                         warm_start=False, n_jobs=1)

##...Fitting the curve.. here we can use both X_train_minmax and X_train_scale separately
log.fit(X_train_scale,Y_train)


##...Checking for accuracy and confidence to validate the model
accuracy = accuracy_score(Y_test,log.predict(X_test_scale))
print (accuracy)

from sklearn import cross_validation
pred_pro = log.predict_proba(X)

array_y_pred = []
array_prob = []
for i in pred_pro:
    if i[0] > i[1]:
        array_y_pred.append(0)
    else:
        array_y_pred.append(1)
    array_prob.append(i[0])

df_final = df
df_final['pred_dv'] = pd.Series(array_y_pred)
df_final['probablity'] = pd.Series(array_prob)
df_final = df_final.sort_values('probablity', ascending=[False])

print (log.coef_)
##print (array_prob)
print df_final[['dv', 'pred_dv', 'probablity']].head(15)


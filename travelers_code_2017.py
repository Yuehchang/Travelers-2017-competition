#preprocessing training and test data
import pandas as pd

#train
train = pd.read_csv('Train.csv')

#delete -1 in target 
train = train[train['cancel'] != -1]

# delete outlier in age
train.describe()
train['ni.age'].quantile(0.99) #0.99 => age=75
train = train[train['ni.age'] <= 80]

#missing value, keep premium for future imputation
train.isnull().sum() #sum(list(dict(train.isnull().sum()).values()))
missing_train_num = ['tenure', 'claim.ind', 'n.adults', 'n.children', 
                       'ni.marital.status', 'len.at.res', 'premium']
missing_train_cat = ['ni.gender', 'sales.channel', 'coverage.type',
                       'dwelling.type', 'credit', 'house.color']

train[missing_train_num] = train[missing_train_num].fillna(train[missing_train_num].mean()) #missing_is_t =  train[train.isnull().any(axis=1)]
train[missing_train_cat] = train[missing_train_cat].fillna('other')

#recode zip for 2 new columns 
#step1 map state by uszipcode
from uszipcode import ZipcodeSearchEngine
search = ZipcodeSearchEngine()

zip_dict = dict(train['zip.code'])
state = []
for index in zip_dict.keys():
    temp = search.by_zipcode('{}'.format(train.loc[index, 'zip.code'].astype(int)))
    temp = temp['State']
    print(index, temp)
    state.append(temp)
print(len(state))

#step2 join state into train
state_list = pd.Series(state)
train['State'] = state_list.values
train['zip.code'] = train['zip.code'].astype(str)
train['State_zip'] = train['zip.code'].str[0:2]
state_dict = dict(zip(train.State_zip, train.State))
state_dict['15'] = 'PA'
state_dict['20'] = 'VA'
state_dict['na'] = 'OTHER'
train['State'] = train.State.fillna(train.State_zip.map(state_dict))
train.State.isnull().sum() #check if there is still null value or not

#create dummy
train.columns
category = ['ni.gender', 'sales.channel', 'coverage.type', 'dwelling.type',
            'house.color', 'State']
train_dummy = pd.DataFrame(index=train.index)
for i in category:
    temp = pd.get_dummies(train[i], prefix=i)
    train_dummy = pd.concat([train_dummy, temp], axis=1)
train = pd.concat([train, train_dummy], axis=1)

##delete dummy n-1
train.columns
drop_dummy = ['ni.gender_other', 'sales.channel_other', 'coverage.type_other', 
              'dwelling.type_other', 'house.color_other', 'State_OTHER']
train = train.drop(drop_dummy, axis=1)

# category to ordinal
ordinal_credit = {'high': 3, 'medium': 2, 'low': 1, 'other': 2}
train['credit_ordinal'] = train['credit'].map(ordinal_credit)

train.to_csv('Train_v5_with_missing.csv')

###############################
#test
test = pd.read_csv('Test.csv')
test.isnull().sum()

missing_test_num = ['tenure', 'claim.ind', 'n.adults', 'ni.age', 
                    'ni.marital.status', 'len.at.res', 'premium']
missing_test_cat = ['ni.gender', 'sales.channel', 'coverage.type',
                    'dwelling.type', 'credit']

test[missing_test_num] = test[missing_test_num].fillna(test[missing_test_num].mean()) 
test[missing_test_cat] = test[missing_test_cat].fillna('other')


zip_dict = dict(test['zip.code'])
state = []
for index in zip_dict.keys():
    temp = search.by_zipcode('{}'.format(test.loc[index, 'zip.code'].astype(int)))
    temp = temp['State']
    print(index, temp)
    state.append(temp)
print(len(state))

state_list = pd.Series(state)
test['State'] = state_list.values
test['zip.code'] = test['zip.code'].astype(str)
test['State_zip'] = test['zip.code'].str[0:2]
state_dict = dict(zip(test.State_zip, test.State))
state_dict['85'] = 'AZ'
state_dict['20'] = 'VA'
state_dict['na'] = 'OTHER'
test['State'] = test.State.fillna(test.State_zip.map(state_dict))
test.State.isnull().sum()

test['dwelling.type'] = test['dwelling.type'].replace(to_replace='Landlord', value='Tenant') 
test['dwelling.type'].value_counts()

test.columns
category = ['ni.gender', 'sales.channel', 'coverage.type', 'dwelling.type',
            'house.color', 'State']
test_dummy = pd.DataFrame(index=test.index)
for i in category:
    temp = pd.get_dummies(test[i], prefix=i)
    test_dummy = pd.concat([test_dummy, temp], axis=1)
test = pd.concat([test, test_dummy], axis=1)

test.columns
drop_dummy = ['ni.gender_other', 'sales.channel_other', 'coverage.type_other', 
              'dwelling.type_other', 'State_OTHER']
test = test.drop(drop_dummy, axis=1)

ordinal_credit = {'high': 3, 'medium': 2, 'low': 1, 'other': 2}
test['credit_ordinal'] = test['credit'].map(ordinal_credit)
test.to_csv('Test_v1_with_missing.csv')

###############################
#model fitting
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv('Train_v5_with_missing.csv', index_col=0)

data.columns

model_columns = ['tenure', 'claim.ind', 'n.adults', 'n.children', 'ni.age',
                 'ni.marital.status', 'premium', 'len.at.res', 'ni.gender_F',
                 'ni.gender_M', 'sales.channel_Broker', 'sales.channel_Online',
                 'sales.channel_Phone', 'coverage.type_A', 'coverage.type_B',
                 'coverage.type_C', 'dwelling.type_Condo', 'dwelling.type_House',
                 'dwelling.type_Tenant', 'house.color_blue', 'house.color_red',
                 'house.color_white', 'house.color_yellow', 'State_AZ', 'State_CO',
                 'State_IA', 'State_PA', 'State_VA', 'State_WA', 'credit_ordinal', 'cancel']

data1 = data[model_columns]

X = data1.iloc[:, 0:-1]
y = data1.iloc[:, -1]


#create training and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

"""
PCA
"""
#scatter plot for PC1 and PC2
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)
        
pca = PCA(n_components=2)
lr = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

"""
Kernel PCA
"""
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X_train)
plt.scatter(X_kpca[y_train==0, 0], X_kpca[y_train==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X_kpca[y_train==1, 0], X_kpca[y_train==1, 1], color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

plot_decision_regions(X_kpca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

"""
Logistic Regression
"""
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(penalty='l2', random_state=1))])
pipe_lr.fit(X_train, y_train)
print('Test Accuracy: {:.3f}'.format(pipe_lr.score(X_test, y_test)))

"""
SVM
"""
pipe_svm = Pipeline([('scl', StandardScaler()),
                     ('clf', SVC(random_state=1))])
pipe_svm.fit(X_train, y_train)
print('Test Accuracy: {:.3f}'.format(pipe_svm.score(X_test, y_test)))

"""
GBC
"""
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0)
gbc.fit(X_train, y_train)
print('Test Accuracy: {:.3f}'.format(gbc.score(X_test, y_test)))

"""
k-fold
"""
clf_list = [pipe_lr, pipe_svm, gbc]
label_list = ['Logistic Regression', 'SVM', 'GBM']
for clf, label in zip(clf_list, label_list):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc',
                             n_jobs=1)
    print('CV accuracy: {0:.3f} +/- {1:.3f} [{2}]\n'.format(np.mean(scores), np.std(scores), label))

"""
GridSearch CV
"""
#gs_svc
pipe_svm.get_params
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0] #there is no 0 inside!!
param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
              {'clf__C': param_range,
               'clf__gamma': param_range,
               'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svm,
                  param_grid=param_grid,
                  scoring='roc_auc',
                  cv=10, 
                  n_jobs=-1,
                  verbose=20)

gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
print(gs.score(X_test, y_test))

#gs_gbm: do not run this model
param_grid_gbm = [{'n_estimators': np.arange(20, 81, 10),
                   'max_depth': np.arange(5,16,2), 
                   'min_samples_split': np.arange(200,1001,200),
                   'min_samples_leaf': np.arange(30,71,10),
                   'max_features': np.arange(7,20,2),
                   'subsample': [0.6,0.7,0.75,0.8,0.85,0.9]}]

gs_gbc = GridSearchCV(estimator=gbc,
                     param_grid=param_grid_gbm,
                     scoring='roc_auc',
                     cv=10, 
                     n_jobs=-1,
                     verbose=20)
gs_gbc = gs_gbc.fit(X_train, y_train)
print(gs_gbc.best_score_)
print(gs_gbc.best_params_)
print(gs_gbc.score(X_test, y_test))

#gs_lr
param_range_lr = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid_lr = [{'clf__C': param_range_lr,
                  'clf__penalty': ['l1', 'l2']}]
gs_lr = GridSearchCV(estimator=pipe_lr,
                     param_grid=param_grid_lr,
                     scoring='roc_auc',
                     cv=10, 
                     n_jobs=-1,
                     verbose=20)
gs_lr = gs_lr.fit(X_train, y_train)
print(gs_lr.best_score_)
print(gs_lr.best_params_)
print(gs_lr.score(X_test, y_test))
coefficients = pd.DataFrame(gs_lr.best_estimator_.named_steps['clf'].coef_, columns=X_train.columns)  
intercept = pd.DataFrame(gs_lr.best_estimator_.named_steps['clf'].intercept_, columns=['intercept'])
print(coefficients)
print(intercept)
all_coef = pd.concat([intercept, coefficients], axis=1)
all_coef.to_csv('first_submisson_Logistic_classifier\'s_coefficients.csv', index=False)

###############################
#running pred

data_test = pd.read_csv('Test_v1_with_missing.csv', index_col=0)

data_test.columns
model_test_columns = ['tenure', 'claim.ind', 'n.adults', 'n.children', 'ni.age',
                      'ni.marital.status', 'premium', 'len.at.res', 'ni.gender_F',
                      'ni.gender_M', 'sales.channel_Broker', 'sales.channel_Online',
                      'sales.channel_Phone', 'coverage.type_A', 'coverage.type_B',
                      'coverage.type_C', 'dwelling.type_Condo', 'dwelling.type_House',
                      'dwelling.type_Tenant', 'house.color_blue', 'house.color_red',
                      'house.color_white', 'house.color_yellow', 'State_AZ', 'State_CO',
                      'State_IA', 'State_PA', 'State_VA', 'State_WA', 'credit_ordinal']
data2 = data_test[model_test_columns]

#best model is logistic Regression
y_prob = gs_lr.predict_proba(data2)
y_prob_1 = y_prob[:, 1]
y_prob_1 = pd.Series(y_prob_1)
y_id = data_test.id
y = pd.concat([y_id, y_prob_1], axis=1)
y.columns = ['id', 'pred']
y.to_csv('prediction_submission.csv', index=False)


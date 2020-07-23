# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 14:11:53 2020

@author: yazılım
"""
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)


import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV,Lasso,Ridge,LassoCV,ElasticNet,ElasticNetCV
from sklearn.linear_model import LassoCV
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR,SVC
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from warnings import filterwarnings
filterwarnings('ignore')


df = pd.read_csv("ornekcsv.csv",sep=";")
df = df.dropna()


dms = pd.get_dummies(df[['soa1', 'soa2', 'soa3']])
y = df["at"]

X_ = df.drop(['at', 'soa1', 'soa2', 'soa3'], axis=1).astype('float64')
X = pd.concat([X_, dms[['soa1', 'soa2', 'soa3']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train.head()
X_test.head()
y_train.head()
y_test.head()


"""
Bütün hataları gösterme
"""

df = pd.read_csv("ornekcsv.csv",sep=";")
df = df.dropna()
dms = pd.get_dummies(df[['soa1', 'soa2', 'soa3']])


def compML(df, y, alg):
    #train-test ayrimi
    y = df[y]
    X_ = df.drop(['au', 'soa1', 'soa2', 'soa3'], axis=1).astype('float64')
    X = pd.concat([X_, dms[['soa1', 'soa2', 'soa3']]], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
    #modelleme
    model = alg().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    model_ismi = alg.__name__
    print(model_ismi, "Modeli Test Hatası:",RMSE)

compML(df, "au", SVR)

models = [LGBMRegressor, 
          XGBRegressor, 
          GradientBoostingRegressor, 
          RandomForestRegressor, 
          DecisionTreeRegressor,
          MLPRegressor,
          KNeighborsRegressor, 
          SVR]


for i in models:
    compML(df, "au", i)


"""
Daha detaylı bulmak için makineyi yoran işlemler
"""

#klasik okuma işlemleri yapılıyor

df = pd.read_csv("diabetes.csv",sep=",")
df = df.dropna()


dms = pd.get_dummies(df[['Age', 'DiabetesPedigreeFunction', 'Insulin']])
y = df["Outcome"]
#okunmayan  değerleri silmem lazım

df



X_ = df.drop(['Outcome','Age', 'DiabetesPedigreeFunction', 'Insulin'], axis=1).astype('float64')
X = pd.concat([X_, dms[['DiabetesPedigreeFunction', 'Insulin']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#model atanıyor
lgb_model = LGBMRegressor().fit(X_train, y_train)


lgb_model


#tahmin yapılıyor

y_pred = lgb_model.predict(X_test)
y_pred
#r^2 li tahmin bunla diğer değeri kıyaslıyorum
np.sqrt(mean_squared_error(y_test, y_pred))


lgb_model = LGBMRegressor()
lgb_model

#parametre ayarlıyoruz
lgbm_params = {"learning_rate": [0.01, 0.1, 0.5, 1, 1.5,2],
              "n_estimators": [20,40,100,200,500,1000],
              "max_depth": [1,3,5,7,9,11,20]}

#makineye en uygun değerler bulmasını söylüyoruz
lgbm_cv_model = GridSearchCV(lgb_model, 
                             lgbm_params, 
                             cv = 10, 
                             n_jobs = -1, 
                             verbose =2).fit(X_train, y_train)


#bulunan sonuçlarda en iyi sonuçlar
lgbm_cv_model.best_params_

#verilen sonuçları yerine koyuyoruz
lgbm_tuned = LGBMRegressor(learning_rate = 0.1, 
                          max_depth = 1, 
                          n_estimators = 100).fit(X_train, y_train)

#yenisi atandı
y_pred = lgbm_tuned.predict(X_test)

#en uygun değerlerle hata oranı yapılıyor
np.sqrt(mean_squared_error(y_test, y_pred))

#
#
#SINIFLANDIRMA MODÜLÜ
#
from sklearn.svm import SVR
df = pd.read_csv("diabetes.csv",sep=",")
df = df.dropna()

y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.30, 
                                                    random_state=42)
svm_model = SVC(kernel = "linear").fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy_score(y_test, y_pred)

svm = SVC()
svm_params = {"C": np.arange(1,10), "kernel": ["linear","rbf"]}
svm_cv_model = GridSearchCV(svm, svm_params, cv = 5, n_jobs = -1, verbose = 2).fit(X_train, y_train)
svm_cv_model.best_score_
svm_cv_model.best_params_
svm_tuned = SVC(C = 2, kernel = "linear").fit(X_train, y_train)
y_pred = svm_tuned.predict(X_test)
accuracy_score(y_test, y_pred)





#nesne temelli olan yer

# Temel Bileşen Analizi

df = pd.read_csv("./Hitters.csv")
df.dropna(inplace = True)
df = df._get_numeric_data()
df.head()

from sklearn.preprocessing import StandardScaler
df = StandardScaler().fit_transform(df)
df[0:5,0:5]
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca_fit = pca.fit_transform(df)
bilesen_df = pd.DataFrame(data = pca_fit, columns = ["birinci_bilesen","ikinci_bilesen"])
bilesen_df
pca.explained_variance_ratio_
pca.components_[1]
#optimum bilese sayisi
pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı");
pca.explained_variance_ratio_
#final
pca = PCA(n_components = 3)
pca_fit = pca.fit_transform(df)
#burası açıklama oranı oluyor bu sayede o bilggilerle ne kadar açıklayıcı olabildiğimizi görebiliyoruz
pca.explained_variance_ratio_

#örenek yapıyorum


df = pd.read_csv("diabetes.csv",sep=",")
df = df.dropna()


dms = pd.get_dummies(df[['Age', 'DiabetesPedigreeFunction', 'Insulin']])
y = df["Outcome"]
#okunmayan  değerleri silmem lazım

df



X_ = df.drop(['Outcome','Age', 'DiabetesPedigreeFunction', 'Insulin'], axis=1).astype('float64')
X = pd.concat([X_, dms[['DiabetesPedigreeFunction', 'Insulin']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#model atanıyor
lgb_model = LGBMRegressor().fit(X_train, y_train)


lgb_model


#tahmin yapılıyor

y_pred = lgb_model.predict(X_test)
y_pred
#r^2 li tahmin bunla diğer değeri kıyaslıyorum
np.sqrt(mean_squared_error(y_test, y_pred))


lgb_model = LGBMRegressor()
lgb_model

#parametre ayarlıyoruz
lgbm_params = {"learning_rate": [0.01, 0.1, 0.5, 1, 1.5,2],
              "n_estimators": [20,40,100,200,500,1000],
              "max_depth": [1,3,5,7,9,11,20]}

#makineye en uygun değerler bulmasını söylüyoruz
lgbm_cv_model = GridSearchCV(lgb_model, 
                             lgbm_params, 
                             cv = 10, 
                             n_jobs = -1, 
                             verbose =2).fit(X_train, y_train)


#bulunan sonuçlarda en iyi sonuçlar
lgbm_cv_model.best_params_

#verilen sonuçları yerine koyuyoruz
lgbm_tuned = LGBMRegressor(learning_rate = 0.1, 
                          max_depth = 1, 
                          n_estimators = 100).fit(X_train, y_train)

#yenisi atandı
y_pred = lgbm_tuned.predict(X_test)

#en uygun değerlerle hata oranı yapılıyor
np.sqrt(mean_squared_error(y_test, y_pred))


from sklearn.svm import SVR
df = pd.read_csv("diabetes.csv",sep=",")
df = df.dropna()

y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.30, 
                                                    random_state=42)
svm_model = SVC(kernel = "linear").fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy_score(y_test, y_pred)

svm = SVC()
svm_params = {"C": np.arange(1,10), "kernel": ["linear","rbf"]}
svm_cv_model = GridSearchCV(svm, svm_params, cv = 5, n_jobs = -1, verbose = 2).fit(X_train, y_train)
svm_cv_model.best_score_
svm_cv_model.best_params_
svm_tuned = SVC(C = 2, kernel = "linear").fit(X_train, y_train)
y_pred = svm_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
# çooook önemli burası bize karşılaştırma sonucunu veriyor
Pregnancies=input("diabet sonuçları:")
Glucose=input("Pregnancies:")
BloodPressure=input("Pregnancies:")
SkinThickness=input("Pregnancies:")
Insulin=input("Pregnancies:")
BMI=input("Pregnancies:")
DiabetesPedigreeFunction=input("Pregnancies:")
Age=input("Pregnancies:")
X_new = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
print("X_new.shape: {}".format(X_new.shape))

prediction = svm_tuned.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(prediction))
#doğruluk oranı aldım şimdi sırada şu değerlere sahip kişi nedir değerini vermede


import pandas as pd
df = pd.read_csv("diabetes.csv")
df = df.dropna()


y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)
"""
dms = pd.get_dummies(df[['Pregnancies', 'Glucose', 'BloodPressure']])
y = df["Outcome"]
"""
x
y
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)








"""
#çok önemli yer sonuç veriyor

X_new = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
print("X_new.shape: {}".format(X_new.shape))

prediction = svm_tuned.predict(X_new)




"""
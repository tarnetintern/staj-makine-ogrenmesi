
"""

Tahmin programları v1


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
from sklearn.svm import SVR

"""

Diyabetli hasta tahmin programı v1


"""

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
"""
svm_params = {"C": np.arange(1,10), "kernel": ["linear","rbf"]}
svm_cv_model = GridSearchCV(svm, svm_params, cv = 5, n_jobs = -1, verbose = 2).fit(X_train, y_train)
svm_cv_model.best_score_
svm_cv_model.best_params_
"""
svm_tuned = SVC(C = 2, kernel = "linear").fit(X_train, y_train)
y_pred = svm_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
# çooook önemli burası bize karşılaştırma sonucunu veriyor
Pregnancies=input("Pregnancies:")
Glucose=input("Glucose:")
BloodPressure=input("BloodPressure:")
SkinThickness=input("SkinThickness:")
Insulin=input("Insulin:")
BMI=input("BMI:")
DiabetesPedigreeFunction=input("DiabetesPedigreeFunction:")
Age=input("Age:")
x_degerler = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
print("x_yeni: {}".format(x_degerler.shape))

prediction = svm_tuned.predict(x_degerler)
print("tahmin: {}".format(prediction))

kontrolcu=int(prediction)
if kontrolcu==0:
    print("Tahmin sonucu: negatif ")
elif kontrolcu==1:
    print("Tahmin sonucu pozitif")


"""


Müşteri analiz programı için tahmin programı lite



"""

from sklearn.svm import SVR
df = pd.read_csv("ornekcsv.csv",sep=",")
df=df.drop(['au', 'at'], axis=1)
df = df.dropna()


y = df["yas"]
X = df.drop(["yas"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.30, 
                                                    random_state=42)

lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
lgb_model = LGBMRegressor()
lgb_model


#lgbm ile yaptım ve oldu
#hata oranı çok düşük
lgbm_params = {"learning_rate": [0.01, 0.1, 0.5, 1],
              "n_estimators": [20,40,100,200,500,1000],
              "max_depth": [1,2,3,4,5,6,7,8,9,10]}
lgbm_cv_model = GridSearchCV(lgb_model, 
                             lgbm_params, 
                             cv = 10, 
                             n_jobs = -1, 
                             verbose =2).fit(X_train, y_train)
lgbm_cv_model.best_params_
lgbm_tuned = LGBMRegressor(learning_rate = 0.01, 
                          max_depth = 3, 
                          n_estimators = 100).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


x_degerler = np.array([["1","3","1"]])
print("hata oranı: %s"%np.sqrt(mean_squared_error(y_test, y_pred)))
print("x_yeni: {}".format(x_degerler.shape))

prediction = lgbm_tuned.predict(x_degerler)
print("tahmin: {}".format(prediction))


"""
        
Müşteri analiz yeni düzen için deneme


"""



from sklearn.svm import SVR
df = pd.read_csv("csvdosyam.csv",sep=",")
#df=df.drop(['au', 'at'], axis=1)
df = df.dropna()


y = df["yas"]
X = df.drop(["yas", "au", "bilgisayar", "telefon", "casual", "sanat","tablet","teknoloji","spor","oyun"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.30, 
                                                    random_state=42)

lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
lgb_model = LGBMRegressor()
lgb_model


#lgbm ile yaptım ve oldu
#hata oranı çok düşük
lgbm_params = {"learning_rate": [0.01, 0.1, 0.5, 1],
              "n_estimators": [20,40,100,200,500,1000],
              "max_depth": [1,2,3,4,5,6,7,8,9,10]}
lgbm_cv_model = GridSearchCV(lgb_model, 
                             lgbm_params, 
                             cv = 10, 
                             n_jobs = -1, 
                             verbose =2).fit(X_train, y_train)
lgbm_cv_model.best_params_
lgbm_tuned = LGBMRegressor(learning_rate = 0.01, 
                          max_depth = 1, 
                          n_estimators = 20).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


x_degerler = np.array([["0"]])
print("hata oranı: %s"%np.sqrt(mean_squared_error(y_test, y_pred)))
print("x_yeni: {}".format(x_degerler.shape))

prediction = lgbm_tuned.predict(x_degerler)
print("tahmin: {}".format(prediction))


#svm için


svm_model = SVC(kernel = "linear").fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy_score(y_test, y_pred)

svm = SVC()

svm_params = {"C": np.arange(1,10), "kernel": ["linear","rbf"]}
svm_cv_model = GridSearchCV(svm, svm_params, cv = 5, n_jobs = -1, verbose = 2).fit(X_train, y_train)
svm_cv_model.best_score_
svm_cv_model.best_params_

svm_tuned = SVC(C = 4, kernel = "linear").fit(X_train, y_train)
y_pred = svm_tuned.predict(X_test)
accuracy_score(y_test, y_pred)



0,0,1,1,1,0,0,1

x_degerler = np.array([["3"]])
print("hata oranı: %s"%np.sqrt(mean_squared_error(y_test, y_pred)))
print("x_yeni: {}".format(x_degerler.shape))

prediction = svm_tuned.predict(x_degerler)
print("tahmin: {}".format(prediction))
















"""

Müşteri analiz programı için tahmin programı
çok uzun
"""
df = pd.read_csv("ornekcsvlite.csv",sep=";")
df = df.dropna()
dms = pd.get_dummies(df[['soa1', 'soa2', 'soa3']])




dms= pd.get_dummies(df[['soa1', 'soa2', 'soa3']])
y= df["yas"]
X_ = df.drop(["yas",'soa1', 'soa2', 'soa3'],axis=1).astype("float64")
X= pd.concat([X_ , dms[['soa1', 'soa2', 'soa3']]], axis=1)
X_train, X_test , y_train , y_test= train_test_split(X,y,test_size=0.25,random_state=42)
df.head()
y_test



svm_model = SVC(kernel = "linear").fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy_score(y_test, y_pred)

svm = SVC()
svm_params = {"C": np.arange(1,10), "kernel": ["linear","rbf"]}
svm_tuned = SVC(C = 2, kernel = "linear").fit(X_train, y_train)
y_pred = svm_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
34,13,1,3,5,174
x_degerler = np.array([["13","1","3","5","174"]])
print("x_yeni: {}".format(x_degerler.shape))

prediction = svm_tuned.predict(x_degerler)
print("tahmin: {}".format(prediction))





"""
Hazır dataset iris dataseti


"""



from sklearn.svm import SVR
df = pd.read_csv("forestfires.csv",sep=",")
df = df.dropna()
df=df.drop(['month', 'day'], axis=1)

y = df["wind"]
X = df.drop(["wind"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.30, 
                                                    random_state=42)



#SVM

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





x_degerler = np.array([["6.3","3.3","6.0","2.5"]])
print("hata oranı: %s"%np.sqrt(mean_squared_error(y_test, y_pred)))
print("x_yeni: {}".format(x_degerler.shape))

prediction = svm_tuned.predict(x_degerler)
print("tahmin: {}".format(prediction))



#LGBM


lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
lgb_model = LGBMRegressor()
lgb_model


#lgbm ile yaptım ve oldu
#hata oranı çok düşük
lgbm_params = {"learning_rate": [0.01, 0.1, 0.5, 1],
              "n_estimators": [20,40,100,200,500,1000],
              "max_depth": [1,2,3,4,5,6,7,8,9,10]}
lgbm_cv_model = GridSearchCV(lgb_model, 
                             lgbm_params, 
                             cv = 10, 
                             n_jobs = -1, 
                             verbose =2).fit(X_train, y_train)
lgbm_cv_model.best_params_
lgbm_tuned = LGBMRegressor(learning_rate = 0.1, 
                          max_depth = 8, 
                          n_estimators = 40).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

92.3,88.9,495.6,8.5,24.1,27,3.1,0,0
x_degerler = np.array([["92.3","88.9","495.6","8.5","24","1","27","3.1","0","0"]])
print("hata oranı: %s"%np.sqrt(mean_squared_error(y_test, y_pred)))
print("x_yeni: {}".format(x_degerler.shape))

prediction = lgbm_tuned.predict(x_degerler)
print("tahmin: {}".format(prediction))






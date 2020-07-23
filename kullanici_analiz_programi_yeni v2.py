# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:55:47 2020

@author: yazılım
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV,Lasso,Ridge,LassoCV,ElasticNet,ElasticNetCV
from sklearn.linear_model import LassoCV
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
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
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.svm import SVR
import csv
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








import pandas as pd
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
def anaMenu():
    
    global menusecim
    
    print("""
  ══════════════════════════════════════════════════════════════
  1-yaş seçimini yapın
  2-seçilen yaş için genel sonuçları ekrana getirin
  3-seçilen yaş için matematiksel sonuçları ekrana getirin
  4-Önce boş csv oluştur ! ilk defa yapacaksan
  5-Sonra csv verilerini güncelle ! ilk defa yapacaksan
  6-Anlizli hata oranları seçilen yaş için
  7-Analizli hata oranları Ana dosya için
  8-Reklam verisi analizi
  9-Çıkış
  
             https://batuhanokmen.com/
     
  ═════════════════════════════════════════════════════════════""")  





def istatistikler_Matematiksel():
    #boş kısımlar 0 yapıldı
    data = pd.read_csv(Secilen_yas,sep=",")
    df = data["yas"].fillna(0)
    df = data["au"].fillna(0)
    df = data["soa1"].fillna(0)
    df = data["soa2"].fillna(0)
    df = data["soa3"].fillna(0)
    df = data["at"].fillna(0)
    
    #son alınan ürünlerin birbiriyle alakası
    x = data[["soa1"]]
    y=data[["soa2"]]
    z=data[["soa3"]]
    reg= LinearRegression()
    model = reg.fit(x,y)
    model_alakasi=model.score(x,y)
    print("Son alınan iki ürünün alakası=%s"%model_alakasi)
    son_alinan_urun_alakasi=model.predict([[1]])
    #ürün alakalırını bulabildim
    if son_alinan_urun_alakasi > float(2):
        print("Alınan ürünlerin alakası yok")
    elif son_alinan_urun_alakasi < float(1.5):
        print("Alınan ürünler yarı yarıya orantılı")
    elif son_alinan_urun_alakasi > float(1.3):
        print("Son alınan ürünler oldukça alakalı")
    else:
        print("Bir hata ile karşılaşıldı, yeniden dene!")
    
        
    

    
    
    
    
    

    auort=np.mean(data["au"])
    print("42 yaşındaki üyeler ortalama %f ürün alırlar" %auort)
    #toplam alışveriş tutarı
    atort=np.mean(data["at"])
    print("Ortalama alışveriş turarları %f dir" %atort)
    #her alışveriş başına harcadıkları ortalama tutar
    averisBasinaTurar=atort / auort
    print("Her ortalama alışverişte %d ₺ harcarlar" % averisBasinaTurar)
    
    
    #beta değerleri bulma
    x = data[["soa1"]]
    y=data[["soa2"]]
    
    
    lm = LinearRegression()
    model = lm.fit(x, y)
    b0=model.intercept_
    b1=model.coef_
    print("Denklemimiz ' y = ß0 +ß1X+e dir ' ")
    print("ß0 = %s"% b0)
    print("ß1= %s" % b1)
    
    #hata oranları
    model.predict(x)
    a=mean_squared_error(x,model.predict(x))
    print("Alınabilecek  ortalama hata oranı %s" %a)
    a2=np.sqrt(a)
    print("Alınabilecek net hata oranı %s" % a2)
    
        
    #sinama yapıma
    

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=99)
    x_train.head()
    x_test.head()
    lm= LinearRegression()
    print("Deneme hataları oranları...")
    #Hata alma 1. yontem 
    #eğitim hataları
    model = lm.fit(x_train,y_train)
    egitim_hata=np.sqrt(mean_squared_error(y_train,model.predict(x_train)))
    print("Eğitim hatası: %s"%egitim_hata)
    #test hataları
    test_hatasi=np.sqrt(mean_squared_error(y_test,model.predict(x_test)))
    print("Test hataları %s"%test_hatasi)
    
    #k katlı cv yöntemi mse
    print("K katlı cv yöntemi ile hata bulumu")
    from sklearn.model_selection import cross_val_score
    cross_val_score(model, x_train,y_train,cv=10 ,scoring="neg_mean_squared_error")
    mse_cv=np.mean(-(cross_val_score(model, x_train,y_train,cv=10 ,scoring="neg_mean_squared_error")))
    print("Test için cv için mse hatası=%s"%mse_cv)
    rmse_cv=np.sqrt(np.mean(-(cross_val_score(model, x_train,y_train,cv=10 ,scoring="neg_mean_squared_error"))))
    print("Test için cv için rmse hatası=%s"%rmse_cv)
    karsilastir_cv=np.sqrt(np.mean(-(cross_val_score(model, x,y,cv=10 ,scoring="neg_mean_squared_error"))))
    print("Test değeri olmadan hata bulumu=%s"%karsilastir_cv)

    












def istatistikler():
    data = pd.read_csv(Secilen_yas,sep=",")
    df=pd.DataFrame(data) 
    belirleyici=data["soa1"]#.append(data["soa2"]).append(data["soa3"])
    a=belirleyici.value_counts() 
    df.describe().T
    print(a)
    liste=list(a.items())
    list1, list2 = zip(*liste)
    birinci_tercih=list1[0]
    kadar_almis=list2[0]
    print("kullanıcı 1. tercih",birinci_tercih,"şu kadar almış",kadar_almis)
    
    
    

    bos_deger=int()
    bilgisayar=int()
    gunluk_giyim=int()
    sanat=int()
    tablet=int()
    teknoloji=int()
    spor=int()
    oyun=int()
    telefon=int()
         
    birinci_kategori=str()
    ikinci_kategori=str()
    ucuncu_kategori=str()
    
    


    sayac=-1
    while 1:
        if sayac <10:
            birinci_kat=list1[0]
            sayac=sayac+1
            if sayac==birinci_kat:
                if sayac==0:
                    birinci_kategori="boş katagori"
                elif birinci_kat==1:
                    birinci_kategori="bilgisayar"
                elif birinci_kat==2:
                    birinci_kategori="telefon"
                elif birinci_kat==3:
                    birinci_kategori="günlük giyim"
                elif birinci_kat==4:
                    birinci_kategori="sanat"
                elif birinci_kat==5:
                    birinci_kategori="tablet"
                elif birinci_kat==6:
                    birinci_kategori="teknoloji"
                elif birinci_kat==7:
                    birinci_kategori="spor"
                elif birinci_kat==8:
                    birinci_kategori="oyun"
                else:
                    print("ayarlandı")
                    sayac=sayac+1
            
        else:
            print("bitti")
            break
        
    
        
    
    

    sayac=-1
    while 1:
        if sayac <10:
            ikinci_kat=list1[1]
            sayac=sayac+1
            if sayac==ikinci_kat:
                print(sayac)
                if sayac==0:
                    ikinci_kategori="boş katagori"
                elif ikinci_kat==1:
                    ikinci_kategori="bilgisayar"
                elif ikinci_kat==2:
                    ikinci_kategori="telefon"
                elif ikinci_kat==3:
                    ikinci_kategori="günlük giyim"
                elif ikinci_kat==4:
                    ikinci_kategori="sanat"
                elif ikinci_kat==5:
                    ikinci_kategori="tablet"
                elif ikinci_kat==6:
                    ikinci_kategori="teknoloji"
                elif ikinci_kat==7:
                    ikinci_kategori="spor"
                elif ikinci_kat==8:
                    ikinci_kategori="oyun"
                else:
                    print("ayarlandı")
                    sayac=sayac+1
            
        else:
            print("bitti")
            break
    sayac=-1
    while 1:
        if sayac <10:
            ucuncu_kat=list1[2]
            sayac=sayac+1
            if sayac==ucuncu_kat:
                if ucuncu_kat==0:
                    ucuncu_kategori="boş katagori"
                elif ucuncu_kat==1:
                    ucuncu_kategori="bilgisayar"
                elif ucuncu_kat==2:
                    ucuncu_kategori="telefon"
                elif ucuncu_kat==3:
                    ucuncu_kategori="günlük giyim"
                elif ucuncu_kat==4:
                    ucuncu_kategori="sanat"
                elif ucuncu_kat==5:
                    ucuncu_kategori="tablet"
                elif ucuncu_kat==6:
                    ucuncu_kategori="teknoloji"
                elif ucuncu_kat==7:
                    ucuncu_kategori="spor"
                elif ucuncu_kat==8:
                    ucuncu_kategori="oyun"
                else:
                    print("ayarlandı")
                    sayac=sayac+1
            
        else:
            print("bitti")
            break
    
    
    
        
    print("Seçmiş olduğunuz kullanıcı profili %s yaş gurubu" % yasSec )
    print("Bu yaş gurubu için bir daha ki alışveriş tahminleri şunlardır...")
    print("Alacağı 1. ürün kategorisi=%s"%birinci_kategori)
    print("Alacağı 2. ürün kategorisi=%s"%ikinci_kategori)
    print("Alacağı 3. ürün kategorisi=%s"%ucuncu_kategori)
    ort_au=np.mean(data["au"])
    print("Ortalama alacağı tek seferde ürün sayısı:%f"%ort_au)
    ort_at=np.mean(data["at"])
    print("Tek seferde yapacağı alışveriş turarı:%f"%ort_at)
    ort_a_basina_harcama=np.sum(data["at"])/ np.sum(data["au"])
    print("Ortalama her ürüne: %f ₺ harcanır"%ort_a_basina_harcama)
    ort_harcama_basina_a=np.sum(data["au"])/ np.sum(data["at"])
    print("her bir birim ürün alışverişinde ortalama: %f ₺ harcanır"%ort_harcama_basina_a)
    toplamHarcama=np.sum(data["at"])
    print("toplam harcanan tutar: %f₺"%toplamHarcama)
    toplamUrun=np.sum(data["au"])
    print("toplam alınan birim ürün: %f"%toplamUrun)










def Boş_csv_olustur():
    print("boş sayfalar oluşturuluyor")
    yas_sayaci=14
    while 1:
        if yas_sayaci <61:


            dAdı=("yas_no%s.csv" % yas_sayaci)
            data = pd.read_csv("ornekcsv.csv",sep=";")
            with open(dAdı, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["yas", "au", "bilgisayar", "telefon", "casual", "sanat","tablet","teknoloji","spor","oyun","at"])
                    yas_sayaci=yas_sayaci+1
                    print("%s oluşturuldu" % yas_sayaci)
        else:
            break
        
        
def cvsVerileriniGir():   
    data = pd.read_csv("csvdosyamUZUN.csv",sep=",")
    sayac=0
    sayac1=1
    yas_sayaci=1
    df=pd.DataFrame(data) 
    a=df[0:1]
    df=pd.DataFrame(data)
    
    
    while 1:
        data = pd.read_csv("csvdosyamUZUN.csv",sep=",")
       
        belirleyici=int(a["yas"])
        yas_sayaci=yas_sayaci+1
        if (sayac1 == 681):
            break
        elif (yas_sayaci > 63):
            yas_sayaci=0
        elif ( belirleyici == yas_sayaci):
            dAdı=("yas_no%s.csv" % yas_sayaci)
            with open(dAdı, 'a', newline='') as file:
                writer = csv.writer(file)
                a=df[sayac:sayac1]
                yas=int(a["yas"])
                au=int(a["au"])
                bilgisayar=int(a["bilgisayar"])
                telefon=int(a["telefon"])
                casual=int(a["casual"])
                sanat=int(a["sanat"])
                tablet=int(a["tablet"])
                teknoloji=int(a["teknoloji"])
                spor=int(a["spor"])
                oyun=int(a["oyun"])
                at=int(a["at"])
  
                
                writer.writerow([yas, au, bilgisayar, telefon, casual, sanat,tablet,teknoloji,spor,oyun,at])
                print("%s yaşında ki kişiler giriliyor" % yas_sayaci)
                sayac=sayac+1
                sayac1=sayac1+1
                yas_sayaci=yas_sayaci+1
                a=df[sayac:sayac1]
    print("bitti")
    
def yasSecim():
    global Secilen_yas
    global yasSec
    global yas
    yasSec=int(input("yaş seçiniz:"))
    Secilen_yas=("yas_no%s.csv"% yasSec)    


def DahaAnalizliSonuclar():    
    df = pd.read_csv(Secilen_yas,sep=",")
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

def DahaAnalizliSonuclarAnaDosya():    
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



def ReklamOneri():
    df = pd.read_csv("ornekcsv.csv",sep=";")
    df=df.drop(['au','at'], axis=1)
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
    
    """
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
    """
    lgbm_tuned = LGBMRegressor(learning_rate = 0.01, 
                              max_depth = 3, 
                              n_estimators = 100).fit(X_train, y_train)
    y_pred = lgbm_tuned.predict(X_test)
    np.sqrt(mean_squared_error(y_test, y_pred))
    
    
    birinci_kategori_secim=input("1. değer giriniz:")
    ikinci_kategori_secim=input("2. değer giriniz:")
    ucuncu_kategori_secim=input("3. değer giriniz:")
    if birinci_kategori_secim=="boş katagori":
        birinci_kategori_secim=int(0)
    elif birinci_kategori_secim=="bilgisayar":
        birinci_kategori_secim=int(1)
    elif birinci_kategori_secim=="telefon":
        birinci_kategori_secim=int(2)
    elif birinci_kategori_secim=="günlük_giyim":
        birinci_kategori_secim=int(3)
    elif birinci_kategori_secim=="sanat":
        birinci_kategori_secim=int(4)
    elif birinci_kategori_secim=="tablet":
        birinci_kategori_secim=int(5)
    elif birinci_kategori_secim=="teknoloji":
        birinci_kategori_secim=int(6)
    elif birinci_kategori_secim=="spor":
        birinci_kategori_secim=int(7)
    elif birinci_kategori_secim=="oyun":
        birinci_kategori_secim=int(8)
        
    if ikinci_kategori_secim=="boş katagori":
        ikinci_kategori_secim=int(0)
    elif ikinci_kategori_secim=="bilgisayar":
        ikinci_kategori_secim=int(1)
    elif ikinci_kategori_secim=="telefon":
        ikinci_kategori_secim=int(2)
    elif ikinci_kategori_secim=="günlük_giyim":
        ikinci_kategori_secim=int(3)
    elif ikinci_kategori_secim=="sanat":
        ikinci_kategori_secim=int(4)
    elif ikinci_kategori_secim=="tablet":
        ikinci_kategori_secim=int(5)
    elif ikinci_kategori_secim=="teknoloji":
        ikinci_kategori_secim=int(6)
    elif ikinci_kategori_secim=="spor":
        ikinci_kategori_secim=int(7)
    elif ikinci_kategori_secim=="oyun":
        ikinci_kategori_secim=int(8)
        
    if ucuncu_kategori_secim=="boş katagori":
        ucuncu_kategori_secim=int(0)
    elif ucuncu_kategori_secim=="bilgisayar":
        ucuncu_kategori_secim=int(1)
    elif ucuncu_kategori_secim=="telefon":
        ucuncu_kategori_secim=int(2)
    elif ucuncu_kategori_secim=="günlük_giyim":
        ucuncu_kategori_secim=int(3)
    elif ucuncu_kategori_secim=="sanat":
        ucuncu_kategori_secim=int(4)
    elif ucuncu_kategori_secim=="tablet":
        ucuncu_kategori_secim=int(5)
    elif ucuncu_kategori_secim=="teknoloji":
        ucuncu_kategori_secim=int(6)
    elif ucuncu_kategori_secim=="spor":
        ucuncu_kategori_secim=int(7)
    elif ucuncu_kategori_secim=="oyun":
        ucuncu_kategori_secim=int(8)
    
        
    
    x_degerler = np.array([[birinci_kategori_secim,ikinci_kategori_secim,ucuncu_kategori_secim]])
    print("hata oranı: %s"%np.sqrt(mean_squared_error(y_test, y_pred)))
    print("x_yeni: {}".format(x_degerler.shape))
    
    prediction = lgbm_tuned.predict(x_degerler)
    print("tahmin: {}".format(prediction))
    print("seçilen katagoriler için önerilen reklam grubu yaşı:%s"%prediction)
    




while True:
    anaMenu()
    secim= int(input("seçim yapınız"))
    if secim == 1:
        yasSecim()
    elif secim == 2:
        istatistikler()        
    elif secim == 3:
        istatistikler_Matematiksel()
    elif secim == 4:
        Boş_csv_olustur()
    elif secim == 5:
        cvsVerileriniGir()
    elif secim == 6:
        DahaAnalizliSonuclar()
    elif secim == 7:
        DahaAnalizliSonuclarAnaDosya()
    elif secim == 8:
        ReklamOneri()
    elif secim == 9:
        break
    else:
        print("Bir hata oluştu")
        
        

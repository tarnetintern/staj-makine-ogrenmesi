# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:06:41 2020
batuhan ökmen
@author: okmen
"""

def anaMenu():
    
    global menusecim
    
    print("""
  ═════════════════════════════════════════════════════

             https://batuhanokmen.com/
      

  ═════════════════════════════════════════════════════ """)  



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

import pandas as pd
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split





import sys


from PyQt5 import QtWidgets,uic,QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLineEdit, QTextBrowser, QPushButton, QVBoxLayout, QProgressBar,QComboBox
from Kullanıcı_analiz_programi_dizayn_python import Ui_MainWindow
from PyQt5.QtCore import QBasicTimer
from PyQt5.QtGui import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import *
from PyQt5.uic import loadUi
import os
import speech_recognition as sr
from gtts import gTTS
import socket
import getpass
import googletrans
from googletrans import Translator
import webbrowser



        
def ceviriProgramim():
    
    class MainWindow(QMainWindow):
        def __init__(self):
            super(MainWindow, self).__init__()
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)
            
            self.ui.comboBox.currentTextChanged.connect(self.selectionchange)

            
        def selectionchange(self):
            secenek=self.ui.comboBox.currentIndex()
            print(secenek)

            ana_dosya_isimi=self.ui.textEdit_2.toPlainText()
            ana_dosya_isimi=("%s.csv"%ana_dosya_isimi)
            yasSec=self.ui.textEdit.toPlainText()
            Secilen_yas=("yas_no%s.csv"% yasSec)  
            
            if secenek == 1:
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
                
                
                
                    
                a1=("Seçmiş olduğunuz kullanıcı profili %s yaş gurubu" % yasSec )
                a2=("Bu yaş gurubu için bir daha ki alışveriş tahminleri şunlardır...")
                a3=print("Alacağı 1. ürün kategorisi=%s"%birinci_kategori)
                a4=print("Alacağı 2. ürün kategorisi=%s"%ikinci_kategori)
                a5=print("Alacağı 3. ürün kategorisi=%s"%ucuncu_kategori)
                ort_au=np.mean(data["au"])
                a6= print("Ortalama alacağı tek seferde ürün sayısı:%f"%ort_au)
                ort_at=np.mean(data["at"])
                a7= print("Tek seferde yapacağı alışveriş turarı:%f"%ort_at)
                ort_a_basina_harcama=np.sum(data["at"])/ np.sum(data["au"])
                a8= print("Ortalama her ürüne: %f ₺ harcanır"%ort_a_basina_harcama)
                ort_harcama_basina_a=np.sum(data["au"])/ np.sum(data["at"])
                a9=print("her bir birim ürün alışverişinde ortalama: %f ₺ harcanır"%ort_harcama_basina_a)
                toplamHarcama=np.sum(data["at"])
                a10=print("toplam harcanan tutar: %f₺"%toplamHarcama)
                toplamUrun=np.sum(data["au"])
                a11=print("toplam alınan birim ürün: %f"%toplamUrun)
                self.ui.textEdit_3.clear()
                self.ui.textEdit_3.append("Seçmiş olduğunuz kullanıcı profili %s yaş gurubu" % yasSec )
                self.ui.textEdit_3.append("Bu yaş gurubu için bir daha ki alışveriş tahminleri şunlardır..." )
                self.ui.textEdit_3.append("Alacağı 1. ürün kategorisi=%s"%birinci_kategori )
                self.ui.textEdit_3.append( "Alacağı 2. ürün kategorisi=%s"%ikinci_kategori)
                self.ui.textEdit_3.append("Alacağı 3. ürün kategorisi=%s"%ucuncu_kategori )
                self.ui.textEdit_3.append("Ortalama alacağı tek seferde ürün sayısı:%f"%ort_au )
                self.ui.textEdit_3.append( "Tek seferde yapacağı alışveriş turarı:%f"%ort_at)
                self.ui.textEdit_3.append( "Ortalama her ürüne: %f ₺ harcanır"%ort_a_basina_harcama)
                self.ui.textEdit_3.append("her bir birim ürün alışverişinde ortalama: %f ₺ harcanır"%ort_harcama_basina_a )
                self.ui.textEdit_3.append( "toplam harcanan tutar: %f₺"%toplamHarcama)
                self.ui.textEdit_3.append("toplam alınan birim ürün: %f"%toplamUrun )

            elif (secenek == 2):
                self.ui.textEdit_3.clear()
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
                self.ui.textEdit_3.append("Son alınan iki ürünün alakası=%s"%model_alakasi)
                son_alinan_urun_alakasi=model.predict([[1]])
                #ürün alakalırını bulabildim
                if son_alinan_urun_alakasi > float(2):
                    self.ui.textEdit_3.append("Alınan ürünlerin alakası yok")
                elif son_alinan_urun_alakasi < float(1.5):
                    self.ui.textEdit_3.append("Alınan ürünler yarı yarıya orantılı")
                elif son_alinan_urun_alakasi > float(1.3):
                    self.ui.textEdit_3.append("Son alınan ürünler oldukça alakalı")
                else:
                    self.ui.textEdit_3.append("Bir hata ile karşılaşıldı, yeniden dene!")
                    
            
                auort=np.mean(data["au"])
                self.ui.textEdit_3.append("42 yaşındaki üyeler ortalama %f ürün alırlar" %auort)
                #toplam alışveriş tutarı
                atort=np.mean(data["at"])
                self.ui.textEdit_3.append("Ortalama alışveriş turarları %f dir" %atort)
                #her alışveriş başına harcadıkları ortalama tutar
                averisBasinaTurar=atort / auort
                self.ui.textEdit_3.append("Her ortalama alışverişte %d ₺ harcarlar" % averisBasinaTurar)
                
                
                #beta değerleri bulma
                x = data[["soa1"]]
                y=data[["soa2"]]
                
                
                lm = LinearRegression()
                model = lm.fit(x, y)
                b0=model.intercept_
                b1=model.coef_
                self.ui.textEdit_3.append("Denklemimiz ' y = ß0 +ß1X+e dir ' ")
                self.ui.textEdit_3.append("ß0 = %s"% b0)
                self.ui.textEdit_3.append("ß1= %s" % b1)
                
                #hata oranları
                model.predict(x)
                a=mean_squared_error(x,model.predict(x))
                self.ui.textEdit_3.append("Alınabilecek  ortalama hata oranı %s" %a)
                a2=np.sqrt(a)
                self.ui.textEdit_3.append("Alınabilecek net hata oranı %s" % a2)
                
                    
                #sinama yapıma
                
            
                x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=99)
                x_train.head()
                x_test.head()
                lm= LinearRegression()
                self.ui.textEdit_3.append("Deneme hataları oranları...")
                #Hata alma 1. yontem 
                #eğitim hataları
                model = lm.fit(x_train,y_train)
                egitim_hata=np.sqrt(mean_squared_error(y_train,model.predict(x_train)))
                self.ui.textEdit_3.append("Eğitim hatası: %s"%egitim_hata)
                #test hataları
                test_hatasi=np.sqrt(mean_squared_error(y_test,model.predict(x_test)))
                self.ui.textEdit_3.append("Test hataları %s"%test_hatasi)
                
                #k katlı cv yöntemi mse
                self.ui.textEdit_3.append("K katlı cv yöntemi ile hata bulumu")
                from sklearn.model_selection import cross_val_score
                cross_val_score(model, x_train,y_train,cv=10 ,scoring="neg_mean_squared_error")
                mse_cv=np.mean(-(cross_val_score(model, x_train,y_train,cv=10 ,scoring="neg_mean_squared_error")))
                self.ui.textEdit_3.append("Test için cv için mse hatası=%s"%mse_cv)
                rmse_cv=np.sqrt(np.mean(-(cross_val_score(model, x_train,y_train,cv=10 ,scoring="neg_mean_squared_error"))))
                self.ui.textEdit_3.append("Test için cv için rmse hatası=%s"%rmse_cv)
                karsilastir_cv=np.sqrt(np.mean(-(cross_val_score(model, x,y,cv=10 ,scoring="neg_mean_squared_error"))))
                self.ui.textEdit_3.append("Test değeri olmadan hata bulumu=%s"%karsilastir_cv)
            elif (secenek == 3):
                
                print("boş sayfalar oluşturuluyor")
                yas_sayaci=14
                while 1:
                    if yas_sayaci <61:
                        
                        dAdı=("yas_no%s.csv" % yas_sayaci)
                        data = pd.read_csv(ana_dosya_isimi,sep=";")
                        with open(dAdı, 'a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(["yas", "au", "soa1", "soa2", "soa3", "at"])
                                yas_sayaci=yas_sayaci+1
                                print("%s oluşturuldu" % yas_sayaci)
                    else:
                        break
            elif (secenek ==4):
                data = pd.read_csv(ana_dosya_isimi,sep=";")
                sayac=0
                sayac1=1
                yas_sayaci=1
                df=pd.DataFrame(data) 
                a=df[0:1]
                df=pd.DataFrame(data)
                
                
                while 1:
                    print("başla")
                    data = pd.read_csv(ana_dosya_isimi,sep=";")
                   
                    belirleyici=int(a["yas"])
                    if (sayac1 == 1596):
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
                            soa1=int(a["soa1"])
                            soa2=int(a["soa2"])
                            soa3=int(a["soa3"])
                            at=int(a["at"])            
                            writer.writerow([yas, au, soa1, soa2, soa3, at])
                            print("%s yaşında ki kişiler giriliyor" % yas_sayaci)
                            sayac=sayac+1
                            sayac1=sayac1+1
                            yas_sayaci=yas_sayaci+1
                            a=df[sayac:sayac1]
                print("bitti")
            
                    
            elif (secenek ==5):
                self.ui.textEdit_3.clear()
                self.ui.textEdit_3.append("Yaş dosyan için")
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
                    a=(model_ismi, "Modeli Test Hatası:",RMSE)
                    b=str(a)
                    self.ui.textEdit_3.append(b)
                
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
                    
                    
            elif(secenek==6):
                self.ui.textEdit_3.clear()
                self.ui.textEdit_3.append("Ana dosyan için")
                
                df = pd.read_csv(ana_dosya_isimi,sep=";")
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
                    a=(model_ismi, "Modeli Test Hatası:",RMSE)
                    b=str(a)
                    self.ui.textEdit_3.append(b)
                
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
                
                
                
    
    
    if __name__ == "__main__":
        app = QApplication(sys.argv)
    
        window = MainWindow()
        window.setWindowTitle("Translate program")
        window.show()
    
        sys.exit(app.exec_())



"""
başlayalım

"""


while True:
    anaMenu()
    secim= 1
    if secim == 1:
        ceviriProgramim()
    else:
        print("Bir hata oluştu")












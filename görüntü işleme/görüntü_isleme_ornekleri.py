# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 09:42:36 2020

@author: yazılım
"""
#ders 1

import cv2
import numpy as np
resim = cv2.imread("resim.jpg")
cv2.imwrite("resim_retro.jpg",resim)
cv2.imshow("dizi resmi",resim)
cv2.waitKey(0)
cv2. destroyAllWindows()
type(resim)


#ders 4
b,g,r= cv2.split(resim)
bolge = resim[100:300,200:400]
resim[0:200,0:200]=bolge
cv2.imshow("kesilen resim",bolge)
cv2.waitKey(0)
cv2. destroyAllWindows()
type(resim)

resim = cv2.imread("resim.jpg")
android = cv2.imread("android.jpg")
android_gri= cv2.cvtColor(android,cv2.COLOR_BGR2GRAY)
yukseklik,genislik=android_gri.shape
roi=resim[0:yukseklik,0:genislik]
ret,mask=cv2.threshold(android_gri,10,255,cv2.THRESH_BINARY)
mask_inver=cv2.bitwise_not(mask)
resim_arka= cv2.bitwise_or(roi,roi,mask = mask_inver)
renkli= cv2.add(resim_arka,android)
resim[0:yukseklik,0:genislik]=renkli
cv2.imshow("kesilen resim",resim)
cv2.waitKey(0)
cv2. destroyAllWindows()


#ders 8



resim = cv2.imread("android.jpg")
resim.shape
#büyütme
resim = cv2.pyrUp(resim)
resim = cv2.pyrDown(resim)
resim.shape

cv2.imshow("kesilen resim",resim)
cv2.waitKey(0)
cv2. destroyAllWindows()


#ders 10 

"""
Kamera

"""

import cv2
import numpy as np

kamera= cv2.VideoCapture(0)

while True:
    ret,kare= kamera.read()
    griton= cv2.cvtColor(kare,cv2.COLOR_BGR2GRAY)
    cv2.imshow("video",griton)
    
    
    if cv2.waitKey(60) & 0xFF == ord("q"):
        break
kamera.release()
cv2.destroyAllWindows()


"""
Nesne tanıma

"""
import cv2
import numpy as np

kamera= cv2.VideoCapture(0)


while True:
    ret,kare= kamera.read()
    
    gri_kare = cv2.cvtColor(kare,cv2.COLOR_BGR2GRAY)
    
    nesne = cv2.imread("el.jpg",0)
    
    
    w,h= nesne.shape
    
    res = cv2.matchTemplate(gri_kare,nesne,cv2.TM_CCOEFF_NORMED)
    
    esik_degeri = 0.8
    
    loc=np.where( res > esik_degeri )
    
    
    for n in zip(*loc[::-1]):
        cv2.rectangle(kare,n,(n[0]+h,n[1]+w),(0,255,0),2)
        cv2.putText(kare,"el",(n[0]+10,n[1]),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
    cv2.imshow("ekran", kare)
    
    if cv2.waitKey(60) & 0xFF == ord("q"):
            break
kamera.release()
cv2.destroyAllWindows()
    
    
"""
ders 17
Portre tarzı bir şey

"""



import cv2
import numpy as np

resim = cv2.imread("resim.jpg")

mask= np.zeros(resim.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),dtype=np.float64)
fgdModel = np.zeros((1,65),dtype=np.float64)
100,0,300,300
rect =(100,0,600,5550)
80,0,300,550#agac
cv2.grabCut(resim,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2= np.where((mask ==0) | (mask == 2),0,1).astype(np.uint8)
 
resim = resim * mask2[:,:,np.newaxis]


cv2.imshow("dizi resmi",resim)

cv2.waitKey(0)
cv2. destroyAllWindows()

#

"""
ders 18

keskinleştirme laplacian
resim
"""


import cv2
import numpy as np

resim = cv2.imread("manit.jpg",22111)

laplacian = cv2.Laplacian(resim,cv2.CV_64F)
sobel_dikey = cv2.Sobel(resim,cv2.CV_64F,0,1,ksize= 5)
cv2.imshow("sobel",sobel_dikey)
cv2.imshow("laplacian",laplacian)


cv2.waitKey(0)
cv2. destroyAllWindows()

"""
keskinleştirme ama kamera ile


"""
import cv2
import numpy as np

kamera= cv2.VideoCapture(0)


while True:
    ret,kare= kamera.read()
    
    laplacian = cv2.Laplacian(kare,cv2.CV_64F)
    sobel_dikey = cv2.Sobel(kare,cv2.CV_64F,0,1,ksize= 5)
    kenarlar = cv2.Canny(kare,50,100)
    cv2.imshow("kare", kare)
    cv2.imshow("laplacian", laplacian)
    cv2.imshow("sobey", sobel_dikey)
    cv2.imshow("canny", kenarlar)
    

    
    if cv2.waitKey(60) & 0xFF == ord("q"):
            break
kamera.release()
cv2.destroyAllWindows()



"""
Yüz tanıma


"""
import cv2
import numpy as np

yuz_casc = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
resim = cv2.imread("resim.jpg")
gri_ton = cv2.cvtColor(resim,cv2.COLOR_BGR2GRAY)
yuzler= yuz_casc.detectMultiScale(gri_ton,1.1,4)

print(yuzler)

for (x,y,w,h) in yuzler:
    cv2.rectangle(resim,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow("resim",resim)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""

Yüz tanıma benim yaptığım xml ile




"""

import cv2
import numpy as np

yuz_casc = cv2.CascadeClassifier("myhaar.xml")
resim = cv2.imread("tele.bmp")
gri_ton = cv2.cvtColor(resim,cv2.COLOR_BGR2GRAY)
yuzler= yuz_casc.detectMultiScale(gri_ton,1.1,4)

print(yuzler)

for (x,y,w,h) in yuzler:
    cv2.rectangle(resim,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow("resim",resim)
cv2.waitKey(0)
cv2.destroyAllWindows()





"""
Kamera ile yüz tespiti

"""



import cv2
import numpy as np

kamera= cv2.VideoCapture(0)


while True:
    ret,kare= kamera.read()
    yuz_casc = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    

    gri_ton = cv2.cvtColor(kare,cv2.COLOR_BGR2GRAY)
    
    yuzler= yuz_casc.detectMultiScale(gri_ton,1.1,4)
    
    for (x,y,w,h) in yuzler:
        cv2.rectangle(kare,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("resim",kare)
        
    
    
    if cv2.waitKey(20) & 0xFF == ord("q"):
            break
kamera.release()
cv2.destroyAllWindows()



"""

Vücut taraması


"""

import cv2
import numpy as np


video=cv2.VideoCapture("walkin.mp4")
insan_bulucu= cv2.CascadeClassifier("haarcascade_fullbody.xml")


while True:
    ret,kare= video.read()
    gri_ton = cv2.cvtColor(kare,cv2.COLOR_BGR2GRAY)
    
    bedenler = insan_bulucu.detectMultiScale(gri_ton,1.1,3)
    
    for (x,y,w,h) in bedenler:
        cv2.rectangle(kare,(x,y),(x+w,y+h),(255,0,0),3)
        
    cv2.imshow("video",kare)
        
    
    
    if cv2.waitKey(5) & 0xFF == ord("q"):
            break
kamera.release()
cv2.destroyAllWindows()



"""
Görüntü işleyip kullanabilme


"""
import cv2
import numpy as np
import os



kamera=cv2.VideoCapture(0)
kernel= np.ones((12,12),np.uint8)

isim="uc"

while True:
    ret, kare=kamera.read()
    kesilmis_kare = kare[0:200,0:250]
    kesilmis_kare_gri= cv2.cvtColor(kesilmis_kare,cv2.COLOR_BGR2GRAY)
    kesilmis_kare_HSV= cv2.cvtColor(kesilmis_kare,cv2.COLOR_BGR2HSV)
    
    alt_degerler = np.array([0,20,40])
    ust_dgerler = np.array([40,255,255])
    
renk_filtre_sonuc= cv2.inRange(kesilmis_kare_HSV, alt_degerler , ust_dgerler )
    
renk_filtre_sonuc = cv2.morphologyEx(renk_filtre_sonuc, cv2.MORPH_CLOSE, kernel)

renk_filtre_sonuc = cv2.dilate(renk_filtre_sonuc,kernel,iterations=1)

sonuc = kesilmis_kare.copy()


"""
Kod kısmı burada

"""
import cv2
import numpy as np
import os

kamera = cv2.VideoCapture(0)
kernel = np.ones((12,12),np.uint8)

def resimFarkBul(resim1,resim2):
    resim2= cv2.resize(resim2,(resim1.shape[1],resim1.shape[0]))
    fark_resim= cv2.absdiff(resim1,resim2)
    fark_sayi = cv2.countNonZero(fark_resim)
    print(fark_sayi)
    
    
    
def VeriYukle():
    veri_isimler = []
    veri_resimler = []
    
    Dosyalar = os.listdir("veri/")
    for Dosya in Dosyalar:
        veri_isimler.append(Dosya.replace(".jpg",""))
        veri_resimler.append(cv2.imread("veri/"+Dosya,0))
        
    return veri_isimler,veri_resimler

veri_isimler,veri_resimler=VeriYukle()

    
def sınıflandır(resim,veri_isimler,veri_resimler):
    min_Index = 0
    min_deger = resimFarkBul(resim,veri_resimler[0])
    for t in range(len(veri_isimler)):
        fark_deger= resimFarkBul(resim,veri_resimler[t])
        if(fark_deger<min_deger):
            min_deger=fark_deger
            min_Index=t
    return veri_isimler[min_Index]


while True:
    
    ret, kare = kamera.read()
    kesilmis_kare= kare[0:200,0:250]
    kesilmis_kare_gri = cv2.cvtColor(kesilmis_kare, cv2.COLOR_BGR2GRAY )
    kesilmis_kare_HSV = cv2.cvtColor(kesilmis_kare, cv2.COLOR_BGR2HSV )
    
    alt_deger = np.array([0,20,40])
    ust_dgerler =np.array([40,255,255])
    
    renk_filtre_sonuc= cv2.inRange(kesilmis_kare_HSV, alt_degerler , ust_dgerler )
    renk_filtre_sonuc = cv2.morphologyEx(renk_filtre_sonuc, cv2.MORPH_CLOSE, kernel)
    renk_filtre_sonuc = cv2.dilate(renk_filtre_sonuc,kernel,iterations=1)
    
    sonuc = kesilmis_kare.copy()
    _,cnts,_ = cv2.findContours(renk_filtre_sonuc,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    max_genislik=0
    max_uzunluk=0
    max_Index=-1
    
    for t in range(len(cnts)):
        cnt = cnts[t]
        x,y,w,h=cv2.boundingRect(cnt)
        if (w>max_genislik and h>max_uzunluk):
            max_uzunluk = h
            max_genislik =w
            max_Index=t
        
    
    
    if (len(cnts)>0):
        x,y,w,h=cv2.boundingRect(cnts[max_Index])
        cv2.rectangle(sonuc,(x,y),(x+w,y+h),(0,255,0),2)
        el_resim = renk_filtre_sonuc[y:y+h,x:x+w]
        cv2.imshow("el resim",el_resim)
        print(sınıflandır(el_resim,veri_isimler,veri_resimler))
    
    cv2.imshow("kare",kare)
    cv2.imshow("kesilmis kare",kesilmis_kare)
    cv2.imshow("renk filtresi sonuc",renk_filtre_sonuc)
    cv2.imshow("sonuc",sonuc)
    
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
    
    
    
    
    
    
    








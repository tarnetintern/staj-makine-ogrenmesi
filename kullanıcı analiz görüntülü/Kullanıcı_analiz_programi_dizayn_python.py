# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:/masaüstü/yazılım ile ilgili herşey/staj/Kullanıcı_analiz_programi_dizayn.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1141, 680)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(20, 50, 151, 31))
        self.textEdit.setObjectName("textEdit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 10, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(590, 50, 504, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.comboBox.setFont(font)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(260, 10, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(260, 50, 181, 31))
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_3.setGeometry(QtCore.QRect(30, 130, 1091, 491))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.textEdit_3.setFont(font)
        self.textEdit_3.setObjectName("textEdit_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Yaş seçimini yapın"))
        self.comboBox.setItemText(0, _translate("MainWindow", "seçiniz"))
        self.comboBox.setItemText(1, _translate("MainWindow", "seçilen yaş için genel sonuçları ekrana getirin"))
        self.comboBox.setItemText(2, _translate("MainWindow", "seçilen yaş için matematiksel sonuçları ekrana getirin"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Önce boş csv oluştur ! ilk defa yapacaksan"))
        self.comboBox.setItemText(4, _translate("MainWindow", "Sonra csv verilerini güncelle ! ilk defa yapacaksan"))
        self.comboBox.setItemText(5, _translate("MainWindow", "Anlizli hata oranları seçilen yaş için"))
        self.comboBox.setItemText(6, _translate("MainWindow", "Analizli hata oranları Ana dosya için"))
        self.label_2.setText(_translate("MainWindow", "Ana kaynak dosyası adı"))


# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(909, 810)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_video = QtWidgets.QLabel(self.centralwidget)
        self.label_video.setGeometry(QtCore.QRect(20, 20, 720, 400))
        self.label_video.setText("")
        self.label_video.setObjectName("label_video")
        self.lineEdit_video_file_path = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_video_file_path.setGeometry(QtCore.QRect(100, 470, 281, 27))
        self.lineEdit_video_file_path.setObjectName("lineEdit_video_file_path")
        self.lineEdit_descriptor = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_descriptor.setGeometry(QtCore.QRect(100, 500, 281, 27))
        self.lineEdit_descriptor.setObjectName("lineEdit_descriptor")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 470, 41, 17))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 500, 71, 20))
        self.label_2.setObjectName("label_2")
        self.checkBox_camera = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_camera.setGeometry(QtCore.QRect(590, 470, 81, 16))
        self.checkBox_camera.setObjectName("checkBox_camera")
        self.pushButton_select_path_video = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_select_path_video.setGeometry(QtCore.QRect(400, 470, 151, 23))
        self.pushButton_select_path_video.setObjectName("pushButton_select_path_video")
        self.checkBox_fern = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_fern.setGeometry(QtCore.QRect(590, 490, 96, 22))
        self.checkBox_fern.setObjectName("checkBox_fern")
        self.checkBox_orb = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_orb.setGeometry(QtCore.QRect(590, 510, 96, 22))
        self.checkBox_orb.setObjectName("checkBox_orb")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(30, 540, 101, 17))
        self.label_3.setObjectName("label_3")
        self.label_frame_id = QtWidgets.QLabel(self.centralwidget)
        self.label_frame_id.setGeometry(QtCore.QRect(140, 540, 66, 17))
        self.label_frame_id.setObjectName("label_frame_id")
        self.checkBox_clust = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_clust.setGeometry(QtCore.QRect(590, 550, 151, 41))
        self.checkBox_clust.setObjectName("checkBox_clust")
        self.checkBox_track_orb = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_track_orb.setGeometry(QtCore.QRect(590, 530, 141, 22))
        self.checkBox_track_orb.setObjectName("checkBox_track_orb")
        self.checkBox_Hungarian = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_Hungarian.setGeometry(QtCore.QRect(590, 590, 96, 22))
        self.checkBox_Hungarian.setObjectName("checkBox_Hungarian")
        self.lineEdit_max_frames_to_skip = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_max_frames_to_skip.setGeometry(QtCore.QRect(120, 580, 61, 20))
        self.lineEdit_max_frames_to_skip.setObjectName("lineEdit_max_frames_to_skip")
        self.lineEdit_nkeypoints = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_nkeypoints.setGeometry(QtCore.QRect(280, 580, 61, 20))
        self.lineEdit_nkeypoints.setObjectName("lineEdit_nkeypoints")
        self.lineEdit_dist_tresh = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_dist_tresh.setGeometry(QtCore.QRect(480, 580, 61, 20))
        self.lineEdit_dist_tresh.setObjectName("lineEdit_dist_tresh")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 570, 101, 41))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(200, 580, 81, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(350, 580, 131, 21))
        self.label_6.setObjectName("label_6")
        self.lineEdit_proc_time = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_proc_time.setGeometry(QtCore.QRect(480, 540, 61, 20))
        self.lineEdit_proc_time.setObjectName("lineEdit_proc_time")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(360, 540, 111, 21))
        self.label_7.setObjectName("label_7")
        self.label_fps = QtWidgets.QLabel(self.centralwidget)
        self.label_fps.setGeometry(QtCore.QRect(270, 540, 66, 17))
        self.label_fps.setObjectName("label_fps")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(230, 540, 31, 17))
        self.label_9.setObjectName("label_9")
        self.lineEdit_orb_area = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_orb_area.setGeometry(QtCore.QRect(120, 620, 61, 20))
        self.lineEdit_orb_area.setObjectName("lineEdit_orb_area")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(40, 620, 81, 16))
        self.label_8.setObjectName("label_8")
        self.checkBox_hung_orb = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_hung_orb.setGeometry(QtCore.QRect(590, 620, 151, 21))
        self.checkBox_hung_orb.setObjectName("checkBox_hung_orb")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(200, 620, 81, 16))
        self.label_10.setObjectName("label_10")
        self.lineEdit_brief_size = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_brief_size.setGeometry(QtCore.QRect(280, 620, 61, 20))
        self.lineEdit_brief_size.setObjectName("lineEdit_brief_size")
        self.horizontalSlider_video = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_video.setGeometry(QtCore.QRect(50, 430, 661, 22))
        self.horizontalSlider_video.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_video.setObjectName("horizontalSlider_video")
        self.label_cur_frame = QtWidgets.QLabel(self.centralwidget)
        self.label_cur_frame.setGeometry(QtCore.QRect(730, 430, 47, 13))
        self.label_cur_frame.setText("")
        self.label_cur_frame.setObjectName("label_cur_frame")
        self.checkBox_harris_corn = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_harris_corn.setGeometry(QtCore.QRect(590, 640, 191, 21))
        self.checkBox_harris_corn.setObjectName("checkBox_harris_corn")
        self.checkBox_eco = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_eco.setGeometry(QtCore.QRect(590, 670, 191, 21))
        self.checkBox_eco.setObjectName("checkBox_eco")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 909, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Video"))
        self.label_2.setText(_translate("MainWindow", "Fern_mat"))
        self.checkBox_camera.setText(_translate("MainWindow", "Camera"))
        self.pushButton_select_path_video.setText(_translate("MainWindow", "Select video"))
        self.checkBox_fern.setText(_translate("MainWindow", "Fern"))
        self.checkBox_orb.setText(_translate("MainWindow", "ORB"))
        self.label_3.setText(_translate("MainWindow", "Current frame:"))
        self.label_frame_id.setText(_translate("MainWindow", "      --//--"))
        self.checkBox_clust.setText(_translate("MainWindow", "Clusterization \n"
"for ORB"))
        self.checkBox_track_orb.setText(_translate("MainWindow", "Tracking by ORB"))
        self.checkBox_Hungarian.setText(_translate("MainWindow", "Hungarian"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p>Max count of <br/>frames to skip</p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "Nkeypoints"))
        self.label_6.setText(_translate("MainWindow", "Distance threshold"))
        self.label_7.setText(_translate("MainWindow", "Processing time"))
        self.label_fps.setText(_translate("MainWindow", "--//--"))
        self.label_9.setText(_translate("MainWindow", "FPS"))
        self.label_8.setText(_translate("MainWindow", "ORB area"))
        self.checkBox_hung_orb.setText(_translate("MainWindow", "Hungarian - ORB"))
        self.label_10.setText(_translate("MainWindow", "BRIEF size"))
        self.checkBox_harris_corn.setText(_translate("MainWindow", "Good Features to track"))
        self.checkBox_eco.setText(_translate("MainWindow", "ECO"))


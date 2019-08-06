import os
import sys
import mainwindow
import numpy as np
import cv2
import scipy.io
import scipy
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QTimer, QTime
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QFileDialog
from feature_extract import Feature_Extractor
from bayesclassifier import BayesClassifier
from proceed_frame import Proceed_Frame
from ORB import orb
from Hungarian_Kalman_2 import Tracker
from eco import ECOTracker
class guiApp(QtWidgets.QMainWindow, mainwindow.Ui_MainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.init_data()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('sword v.1')
        self.lineEdit_video_file_path.editingFinished.connect(self.lineedit_video_file_path_editing_finished)
        self.lineEdit_video_file_path.setText('Corsair_LWIR_20.02.19_15.02.11.avi')
        self.lineedit_video_file_path_editing_finished()
        self.lineEdit_descriptor.editingFinished.connect(self.lineedit_descriptor_editing_finished)
        self.lineEdit_descriptor.setText('fern_data_28_01_2019.mat')
        self.lineedit_descriptor_editing_finished()
        self.lineEdit_descriptor.setFocus()
        self.pushButton_select_path_video.clicked.connect(self.pushButton_select_path_video_clicked)
        self.checkBox_camera.toggled.connect(self.checkBox_camera_toggled)
        self.checkBox_fern.toggled.connect(self.checkBox_fern_toggled)
        self.checkBox_orb.toggled.connect(self.checkBox_orb_toggled)
        self.checkBox_clust.toggled.connect(self.checkBox_clust_toggled)
        self.checkBox_harris_corn.toggled.connect(self.checkBox_harris_corn_toggled)
        self.checkBox_track_orb.toggled.connect(self.checkBox_track_orb_toggled)
        self.checkBox_Hungarian.toggled.connect(self.checkBox_Hungarian_toggled)
        self.checkBox_hung_orb.toggled.connect(self.checkBox_hung_orb_toggled)
        self.checkBox_orb.setChecked(True)
        self.checkBox_clust.setChecked(True)
        self.checkBox_eco.toggled.connect(self.checkBox_eco_toggled)
        self.lineEdit_max_frames_to_skip.editingFinished.connect(self.lineEdit_max_frames_to_skip_editing_finished)
        self.lineEdit_dist_tresh.editingFinished.connect(self.lineEdit_dist_tresh_editing_finished)
        self.lineEdit_nkeypoints.editingFinished.connect(self.lineEdit_nkeypoints_editing_finished)
        self.lineEdit_orb_area.editingFinished.connect(self.lineEdit_orb_area_editing_finished)
        self.lineEdit_proc_time.editingFinished.connect(self.lineEdit_proc_time_editing_finished)
        self.lineEdit_brief_size.editingFinished.connect(self.lineEdit_brief_size_editing_finished)
        self.lineEdit_max_frames_to_skip.setText(str(self.max_frames_to_skip))
        self.lineEdit_dist_tresh.setText(str(self.dist_thresh))
        self.lineEdit_orb_area.setText(str(self.size_orb_area))
        self.lineEdit_nkeypoints.setText(str(self.nkeypoints))
        self.lineEdit_proc_time.setText(str(self.proc_time))
        self.lineEdit_brief_size.setText(str(self.brief_size))
        self.horizontalSlider_video.sliderMoved.connect(self.horizontalSlider_video_moved)
        self.label_cur_frame.setText('00:00')
        pass

    def init_data(self):
        self.src_video_path = None
        self.video_len = None
        self.cur_frame_id = 0
        self.tmr = QTimer()
        self.time = QTime()
        self.proc_time = 40
        self.tmr.setInterval(self.proc_time)
        self.tmr.timeout.connect(self.timeout_handler)
        self.tmr.start()
        self.time.start()
        self.fps_period = None
        self.last_timestamp = self.time.elapsed()
        self.video_capturer = None
        self.frame = None
        self.prev_frame = None
        self.frame_draw = None
        self.frame_crop = None
        self.proc_qimage = None
        self.proc_qpixmap = None
        self.amount_of_frames = None
        self.feat_extr = Feature_Extractor()
        self.bay_cl = BayesClassifier()
        self.proceed_frame = Proceed_Frame()
        self.tracker = Tracker(4, 15, 30, 31)
        self.orb = orb()
        self.dist_thresh = 4
        self.max_frames_to_skip = 15
        self.size_orb_area = 30
        self.left_corner = None
        self.right_corner = None
        self.left_corner_real = None
        self.right_corner_real = None
        self.kp_select = None
        self.nkeypoints = 300
        self.brief_size = 31
        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()
        self.width = None
        self.height = None
        self.enable_playing = False
        self.fern_det = False
        self.orb_det = False
        self.harris_det = False
        self.enable_clust = False
        self.enable_tracking_orb = False
        self.enable_tracking_Hungarian = False
        self.enable_tracking_hung_orb = False
        self.enable_eco = False
        self.camera_id = 0
        self.camera_capturer = cv2.VideoCapture(self.camera_id)
        self.camera_frame = None
        self.video_frame = None
        self.video_src = 'video'
        self.clust_kps = []
        self.en_roi = False
        self.bbox_eco = ()
        self.bbox_init_eco = ()
        self.tracker_eco = ECOTracker(True)
        pass

    def lineedit_video_file_path_editing_finished(self):
        if os.path.isfile(self.lineEdit_video_file_path.text()):
            self.src_video_path = os.path.basename(self.lineEdit_video_file_path.text())
            self.video_capturer = cv2.VideoCapture(self.lineEdit_video_file_path.text())
            self.video_capturer.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame_id)
            self.amount_of_frames = self.video_capturer.get(cv2.CAP_PROP_FRAME_COUNT)
            self.enable_playing = False
            self.refresh_image_tracker()
        else:
            print('Error: path to video file is invalid!')
        pass

    def horizontalSlider_video_moved(self):
        self.cur_frame_id = int(self.amount_of_frames * self.horizontalSlider_video.sliderPosition() / 100)
        self.video_capturer.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame_id)
        self.refresh_image_tracker()
        self.horizontalSlider_time(self.cur_frame_id)
        pass

    def horizontalSlider_time(self, cur_frame_id):
        if cur_frame_id > 1500:
            minutes = int(cur_frame_id // 1500)
            seconds = int((cur_frame_id % 1500) // 25)
        else:
            minutes = 0
            seconds = int(cur_frame_id // 25)
        str_seconds = str(seconds)
        str_minutes = str(minutes)
        if seconds < 10:
            str_seconds = '0' + str(seconds)
        if minutes < 10:
            str_minutes = '0' + str(minutes)

        self.label_cur_frame.setText(str_minutes + ':' + str_seconds)
        pass

    def lineEdit_max_frames_to_skip_editing_finished(self):
        self.max_frames_to_skip = int(self.lineEdit_max_frames_to_skip.text())
        self.tracker = Tracker(self.dist_thresh, self.max_frames_to_skip, self.size_orb_area, self.brief_size)
        pass

    def lineEdit_dist_tresh_editing_finished(self):
        self.dist_thresh = int(self.lineEdit_dist_tresh.text())
        self.tracker = Tracker(self.dist_thresh, self.max_frames_to_skip, self.size_orb_area, self.brief_size)
        pass

    def lineEdit_brief_size_editing_finished(self):
        self.brief_size = int(self.lineEdit_brief_size.text())
        self.tracker = Tracker(self.dist_thresh, self.max_frames_to_skip, self.size_orb_area, self.brief_size)
        pass


    def lineEdit_nkeypoints_editing_finished(self):
        self.nkeypoints = int(self.lineEdit_nkeypoints.text())
        pass

    def lineEdit_orb_area_editing_finished(self):
        self.size_orb_area = int(self.lineEdit_orb_area.text())
        self.tracker = Tracker(self.dist_thresh, self.max_frames_to_skip, self.size_orb_area, self.brief_size)
        pass

    def lineEdit_proc_time_editing_finished(self):
        self.proc_time = int(self.lineEdit_proc_time.text())
        self.tracker = Tracker(self.dist_thresh, self.max_frames_to_skip, self.size_orb_area, self.brief_size)
        pass

    def pushButton_select_path_video_clicked(self):
        fname = QFileDialog.getOpenFileName(self, 'Select video file', '')[0]
        if fname is not None:
            self.src_video_path = os.path.basename(fname)
            self.lineEdit_video_file_path.setText(fname)
            self.lineedit_video_file_path_editing_finished()
        pass

    def lineedit_descriptor_editing_finished(self):
        if os.path.isfile(self.lineEdit_descriptor.text()):
            self.feature = self.lineEdit_descriptor.text()
            mat = scipy.io.loadmat(self.feature)
            self.features = mat['features']['x'][0][0]
            self.weights = mat['WEIGHTS']
            self.feat_extr.config_features(self.features)
            self.bay_cl.config_weights(self.weights)
        else:
            print('Error: path to mat file is invalid!')


    def refresh_image_tracker(self):

        if self.video_src == 'camera':
            self.refresh_camera_frame()
            self.frame = self.camera_frame
        if self.video_src == 'video':
            self.refresh_video_frame()
            self.frame = self.video_frame


        if self.frame is not None:
            a = self.frame.shape[1] / 720
            b = self.frame.shape[0] / 400
            self.roi_rect = ()
            if self.left_corner is not None:
                self.en_roi = True
                self.begin = QtCore.QPoint()
                self.end = QtCore.QPoint()
                self.roi_rect = (self.left_corner_real[0], self.left_corner_real[1], self.right_corner_real[0] - self.left_corner_real[0], self.right_corner_real[1] - self.left_corner_real[1])
                if self.bbox_init_eco != self.roi_rect:
                    self.bbox_eco = ()
            if self.enable_playing:
                self.frame_draw = self.frame
                if self.fern_det:
                    self.frame_draw = self.proceed_frame.scan_window(self.frame_draw, self.features, self.weights)
                elif self.harris_det or self.orb_det or self.enable_tracking_orb or self.enable_tracking_Hungarian or self.enable_tracking_hung_orb:
                    self.tracker.Update(self.frame_draw, self.nkeypoints, self.harris_det, self.enable_clust, self.enable_tracking_Hungarian,
                                        self.enable_tracking_orb, self.enable_tracking_hung_orb, self.en_roi, self.roi_rect)
                    self.frame_draw = self.tracker.draw_tracks(self.frame_draw, self.en_roi, self.roi_rect)
                elif self.enable_eco and len(self.roi_rect) != 0:
                    if len(self.frame_draw.shape) == 3:
                        is_color = True
                    else:
                        is_color = False
                        self.frame_draw = self.frame_draw[:, :, np.newaxis]
                    vis = True
                    if len(self.bbox_eco) == 0:
                        self.bbox_eco = self.roi_rect
                        self.bbox_init_eco = self.roi_rect
                        self.tracker_eco.init(self.frame_draw, self.bbox_eco)
                        self.bbox_eco = (self.bbox_eco[0] - 1, self.bbox_eco[1] - 1,
                                self.bbox_eco[0] + self.bbox_eco[2] - 1, self.bbox_eco[1] + self.bbox_eco[3] - 1)
                    elif self.cur_frame_id < int(self.amount_of_frames) - 1:
                        self.bbox_eco = self.tracker_eco.update(self.frame_draw, True, vis)
                    else:
                        self.bbox_eco = self.tracker_eco.update(self.frame_draw, False, vis)
                    frame = self.frame_draw.squeeze()
                    self.frame_draw = cv2.rectangle(frame,
                                          (int(self.bbox_eco[0]), int(self.bbox_eco[1])),
                                          (int(self.bbox_eco[2]), int(self.bbox_eco[3])),
                                          (0, 0, 255),
                                          1)
                self.proc_qpixmap = self.proceed_frame.qpixmap_from_arr(self.frame_draw)
                self.label_video.setPixmap(self.proc_qpixmap)
            if not self.enable_playing:
                self.frame_draw = self.frame
                if self.harris_det or self.enable_tracking_orb or self.enable_tracking_Hungarian or self.enable_tracking_hung_orb:
                    if self.orb:
                        (self.frame_draw, self.clust_kps) = self.orb.orb_desc(self.frame_draw, self.enable_clust, self.nkeypoints, self.brief_size, self.en_roi, self.roi_rect)
                    elif self.harris_det:
                        self.frame_draw = self.orb.harris_corner_det(self.frame_draw, self.nkeypoints, self.size_orb_area)
                self.proc_qpixmap = self.proceed_frame.qpixmap_from_arr(self.frame_draw)
                self.label_video.setPixmap(self.proc_qpixmap)
        pass

    def timeout_handler(self):
        if self.enable_playing:
            self.cur_frame_id += 1
            self.refresh_image_tracker()
            if self.frame is not None:
                self.label_frame_id.setText(str(self.cur_frame_id))
                self.horizontalSlider_video.setSliderPosition(self.cur_frame_id * 100 // self.amount_of_frames)
                self.horizontalSlider_time(self.cur_frame_id)
        cur_time = self.time.elapsed()
        self.fps_period = cur_time - self.last_timestamp
        self.last_timestamp = cur_time

        if self.fps_period != 0:
            self.label_fps.setText(str(int(1000.0 / self.fps_period)))
        pass

    def refresh_camera_frame(self):
        ret, self.camera_frame = self.camera_capturer.read()
        pass

    def refresh_video_frame(self):
        self.video_capturer.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame_id)
        ret, self.video_frame = self.video_capturer.read()
        pass

    def checkBox_camera_toggled(self, checked):
        if checked:
            self.video_src = 'camera'
            self.refresh_image_tracker()
        else:
            self.video_src = 'video'
            self.refresh_image_tracker()
        pass

    def checkBox_fern_toggled(self, checked):
        self.fern_det = checked
        self.checkBox_orb.setChecked(False)
        pass

    def checkBox_orb_toggled(self, checked):
        self.orb_det = checked
        self.checkBox_fern.setChecked(False)
        pass

    def checkBox_eco_toggled(self, checked):
        self.enable_eco = checked
        self.checkBox_orb.setChecked(False)
        self.checkBox_clust.setChecked(False)
        pass

    def checkBox_harris_corn_toggled(self, checked):
        self.harris_det = checked
        self.checkBox_orb.setChecked(False)
        self.checkBox_clust.setChecked(False)
        pass

    def checkBox_clust_toggled(self, checked):
        self.enable_clust = checked
        pass

    def checkBox_track_orb_toggled(self, checked):
        self.enable_tracking_orb = checked
        self.checkBox_Hungarian.setChecked(False)
        self.checkBox_hung_orb.setChecked(False)
        self.refresh_image_tracker()
        self.tracker.del_tracks()
        pass

    def checkBox_hung_orb_toggled(self, checked):
        self.enable_tracking_hung_orb = checked
        self.checkBox_Hungarian.setChecked(False)
        self.checkBox_track_orb.setChecked(False)
        self.refresh_image_tracker()
        self.tracker.del_tracks()
        pass

    def checkBox_Hungarian_toggled(self, checked):
        self.enable_tracking_Hungarian = checked
        self.checkBox_track_orb.setChecked(False)
        self.checkBox_hung_orb.setChecked(False)
        self.refresh_image_tracker()
        self.tracker.del_tracks()
        pass

    def paintEvent(self, event):
        qp = QPainter()
        pen = QPen(QtCore.Qt.red, 1)
        width = self.end.x() - self.begin.x()
        height = self.end.y() - self.begin.y()
        if self.frame is not None:
            qp.begin(self.label_video.pixmap())
            qp.setPen(pen)
            qp.drawRect(self.begin.x() - 20, self.begin.y() - 20, width, height)
            qp.end()

    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = event.pos()
        self.update()

        pass

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()
        if self.frame is not None:
            a = self.frame.shape[1] / 720
            b = self.frame.shape[0] / 400
            if 20 < self.begin.x() < 740 and 20 < self.end.x() < 740 and 20 < self.begin.y() < 420 and 20 < self.end.y() < \
                    420 and self.end.x() != self.begin.x() and self.end.y() != self.begin.y() and self.enable_playing == False:
                self.left_corner = (self.begin.x() - 20, self.begin.y() - 20)
                self.right_corner = (self.end.x() - 20, self.end.y() - 20)
                self.left_corner_real = (int(a * (self.begin.x() - 20)), int(b * (self.begin.y() - 20)))
                self.right_corner_real = (int(a * (self.end.x() - 20)), int(b * (self.end.y() - 20)))
                self.en_roi = True
                self.roi_rect = (self.left_corner_real[0], self.left_corner_real[1],
                                 self.right_corner_real[0] - self.left_corner_real[0],
                                 self.right_corner_real[1] - self.left_corner_real[1])
                self.proc_qpixmap = self.proceed_frame.qpixmap_from_arr_mouse_move(self.frame, self.left_corner,
                                                                                   self.right_corner)
                self.label_video.setPixmap(self.proc_qpixmap)

    def mouseReleaseEvent(self, event):
        self.end = event.pos()
        self.update()
        if self.frame is not None:
            a = self.frame.shape[1] / 720
            b = self.frame.shape[0] / 400
            self.kp_select = (int(a * (self.end.x() - 20)), int(b * (self.end.y() - 20)))
            if self.kp_select is not None:
                if self.enable_tracking_Hungarian or self.enable_tracking_orb or self.enable_tracking_hung_orb:
                    (self.frame, self.clust_kps) = self.orb.orb_desc(self.frame, self.enable_clust,
                                                                          self.nkeypoints, self.brief_size, self.en_roi, self.roi_rect)
                    if self.en_roi:
                        self.kp_select = (int(a * (self.end.x() - 20)) - self.roi_rect[0], int(b * (self.end.y() - 20)) - self.roi_rect[1])
                    if self.kp_select[0] > 0 and self.kp_select[1] > 0:
                        self.tracker.init_track(self.kp_select, self.clust_kps)
                    self.checkBox_orb.setChecked(False)
        pass

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            self.enable_playing = not self.enable_playing
        if event.key() == QtCore.Qt.Key_F12:
            self.left_corner_real = (0, 0)
            self.right_corner_real = (self.video_frame.shape[1], self.video_frame.shape[0])
        pass



def main():
    app = QtWidgets.QApplication(sys.argv)
    window = guiApp()
    window.show()
    sys.exit(app.exec_())
    pass

if __name__ == '__main__':
    main()
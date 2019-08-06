from bayesclassifier import BayesClassifier
from feature_extract import Feature_Extractor
import numpy as np
import cv2
from PyQt5.QtGui import QImage, qRgb, QPixmap
gray_color_table = [qRgb(i, i, i) for i in range(256)]
class Proceed_Frame:
    def __init__(self):
        self.frame_draw = None
        self.proc_qimage = None
        self.proc_qpixmap = None
        self.amount_of_frames = None

        self.min_win = 8
        self.min_var = 300
        self.thr_fern = 0.5
        self.num_det = 200
        self.shift = 2
        self.feat_extr = Feature_Extractor()
        self.bay_cl = BayesClassifier()
        self.features = None
        self.weights = None
        self.clust_kps = None
        pass

    def scan_window(self, frame, features, weights):
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_draw = frame_gray
        rect_dt = []
        conf_frame = []
        for i in np.int32(np.arange(0, frame_draw.shape[0] - self.min_win +self.shift, self.shift)):
            for j in np.int32(np.arange(0, frame_draw.shape[1] - self.min_win + self.shift, self.shift)):
                roi = frame_draw[i: i + self.min_win, j: j + self.min_win]
                descs = self.feat_extr.fern_desc(roi, features)
                bbox = [j, i, j + self.min_win - 1, i + self.min_win - 1]
                conf = self.bay_cl.predict(descs, weights, roi)
                if conf > self.thr_fern:
                    if np.var(roi) > self.min_var:
                        rect_dt.append(bbox)
                        conf_frame.append(conf)
        if len(conf_frame) > int(self.num_det * frame_draw.size / 172800):
            conf_frame_arr = np.array(conf_frame)
            sIdx = np.argsort(conf_frame_arr)[::-1]
            rect_dt = np.array(rect_dt)
            rect_dt = rect_dt[sIdx[0:int((self.num_det * frame_draw.size / 172800))]]

        for i in np.arange(len(rect_dt)):
            frame_draw = cv2.rectangle(frame, (rect_dt[i][0], rect_dt[i][1]),
                                            (rect_dt[i][2], rect_dt[i][3]), (0, 255, 0), 1)
        return frame_draw


    def qpixmap_from_arr(self, arr):
        if len(arr.shape) == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

        frame = cv2.resize(arr, (720, 400), interpolation=cv2.INTER_CUBIC)
        proc_qimage = self.convert_ndarr_to_qimg(frame)
        proc_qpixmap = QPixmap.fromImage(proc_qimage)
        return proc_qpixmap

    def qpixmap_from_arr_mouse_move(self, arr, left_corner, right_corner):
        frame_draw = arr
        if len(arr.shape) == 3:
            frame_draw = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
        frame_draw = cv2.resize(frame_draw, (720, 400), interpolation=cv2.INTER_CUBIC)
        frame_label = frame_draw
        frame_label = cv2.rectangle(frame_label, left_corner, right_corner, (0, 0, 255), 1)
        proc_qimage = self.convert_ndarr_to_qimg(frame_label)
        proc_qpixmap = QPixmap.fromImage(proc_qimage)
        return proc_qpixmap


    def convert_ndarr_to_qimg(self,arr):
        if arr is None:
            return QImage()
        qim = None
        if arr.dtype is not np.uint8:
            arr = arr.astype(np.uint8)
        if arr.dtype == np.uint8:
            if len(arr.shape) == 2:
                qim = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_Indexed8)
                qim.setColorTable(gray_color_table)
            elif len(arr.shape) == 3:
                if arr.shape[2] == 3:
                    qim = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGB888)
        return qim.copy()

import numpy as np
import cv2
from proceed_frame import Proceed_Frame

class orb:

    def __init__(self):

        self.clust_kps = None
        self.proceed_frame = Proceed_Frame()
        self.kps_sort = None
        self.frame_init = None
        pass

    def orb_desc(self, frame, enable_clust, nkps, brief_size, en_roi, roi_rect):
        img = frame
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if en_roi:
            img = img[roi_rect[1]:roi_rect[1] + roi_rect[3], roi_rect[0]:roi_rect[0] + roi_rect[2]]
        orb = cv2.ORB_create(nfeatures=nkps, edgeThreshold=brief_size, patchSize=brief_size)
        kp = orb.detect(img, None)
        if enable_clust:
            self.clust_kps = self.clust_keypoints(kp)
            img2 = cv2.drawKeypoints(frame, self.clust_kps, outImage=None, color=(0, 255, 0), flags = 0)
        else:
            self.clust_kps = kp
            img2 = cv2.drawKeypoints(frame, kp, outImage=None, color=(0, 255, 0), flags=0)
        frame_draw = img2
        return (frame_draw, self.clust_kps)

    def clust_keypoints(self, keypoints):
        if len(keypoints) != 0:
            coord_arr = np.array([keypoints[0].pt])
            kp_array = np.array(keypoints)
            self.clust_kps = np.array([])
            for i in range(1, len(keypoints)):
                coord_arr = np.concatenate([coord_arr, [keypoints[i].pt]])
            for i in range(coord_arr.shape[0]):
                if len(coord_arr) == 0:
                    break
                cur_kp = np.tile(coord_arr[0], (coord_arr.shape[0], 1))
                distance = np.sqrt((coord_arr[:, 0] - cur_kp[:, 0])**2 + (coord_arr[:, 1] - cur_kp[:, 1])**2)
                distance_sort = np.sort(distance)
                indices_sort = np.argsort(distance)
                distance_clust = distance_sort[distance_sort < 4]
                indices_clust = indices_sort[:distance_clust.size]
                self.clust_kps = np.concatenate([self.clust_kps, [kp_array[0]]])
                coord_arr = np.delete(coord_arr, indices_clust, axis=0)
                kp_array = np.delete(kp_array, indices_clust)
        return self.clust_kps


    def harris_corner_det(self, img, nkps, euc_dist):
        img_har = img
        if len(img.shape) == 3:
            img_har = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(img_har, nkps, 0.01, euc_dist, mask=np.array([]), blockSize=3, useHarrisDetector=0,
                                          k=0.04)
        for i in range(corners.shape[0]):
            img = cv2.circle(img,(int(corners[i][0][0]), int(corners[i][0][1])),
                              4, (255, 255, 0), 2)
        return img

    def track_kp_selected(self, img, nkps, dist_tresh, kp, distance, kp_pred, brief_size):
        orb = cv2.ORB_create(nfeatures=nkps, edgeThreshold=brief_size, patchSize=brief_size)
        if kp is not None:
            kp_array = np.array(kp)
            indices = np.argsort(distance)
            distance_sort = np.sort(distance)
            distance_area = distance_sort[distance_sort < dist_tresh]
            indices_area = indices[:distance_area.size]
            kp_array = kp_array[indices_area]
            kp2 = list(kp_array)
            if len(kp2) != 0:
                kp2, des2 = orb.compute(img, kp2)
                [kp_pred], des1 = orb.compute(img, [kp_pred])
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                match = bf.match(des1, des2)
                return [indices_area[match[0].trainIdx], match[0].distance]
            else:
                return [0, 500]


    def init_orb(self, img, kp_select):
        if img is not None:
            self.frame_init = img
        if self.clust_kps is not None:
            coord_arr = np.array([self.clust_kps[0].pt])
            for i in range(1, len(self.clust_kps)):
                coord_arr = np.concatenate([coord_arr, [self.clust_kps[i].pt]])
            cur_kp = np.tile(kp_select, (coord_arr.shape[0], 1))
            distance = np.sqrt((coord_arr[:, 0] - cur_kp[:, 0]) ** 2 + (coord_arr[:, 1] - cur_kp[:, 1]) ** 2)
            indices_sort = np.argsort(distance)
            self.kps_sort = self.clust_kps[indices_sort]

        pass


import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filter_2 import KalmanFilter
from ORB import orb

class Track(object):

    def __init__(self, kp):
        self. prediction = np.array([[kp.pt[0]], [kp.pt[1]]])
        self.skipped_frames = 0
        self.kp = kp
        self.KF = KalmanFilter()
        self.invisible_count = 0
        self.distance_Hamm = []
        self.key_point = kp
        pass

class Tracker(object):

    def __init__(self, dist_thresh, max_frames_to_skip, size_orb_area, brief_size):
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.tracks = []
        self.kps_sort = None
        self.orb = orb()
        self.size_orb_area = size_orb_area
        self.brief_size = brief_size
        pass

    def init_track(self, kp_select, clust_kps):
        indices_sort = []
        if clust_kps is not None:
            coord_arr = np.array([clust_kps[0].pt])
            for i in range(1, len(clust_kps)):
                coord_arr = np.concatenate([coord_arr, [clust_kps[i].pt]])
            cur_kp = np.tile(kp_select, (coord_arr.shape[0], 1))
            distance = np.sqrt((coord_arr[:, 0] - cur_kp[:, 0]) ** 2 + (coord_arr[:, 1] - cur_kp[:, 1]) ** 2)
            indices_sort = np.argsort(distance)
        new_track = Track(clust_kps[indices_sort[0]])
        self.tracks.append(new_track)
        pass

    def del_tracks(self):
        self.tracks = []
        pass


    def Update(self, img, nkps, harris_det, enable_clust,  tracker_hungarian, tracker_orb, tracker_conj_orb_hung, en_roi, roi_rect):
        img2 = img
        if len(img.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        if en_roi:
            img2 = img2[roi_rect[1]:roi_rect[1] + roi_rect[3], roi_rect[0]:roi_rect[0] + roi_rect[2]]
        if harris_det:
            orb = cv2.ORB_create(nfeatures=nkps, edgeThreshold=self.brief_size, patchSize=self.brief_size)
            corners = cv2.goodFeaturesToTrack(img2, nkps, 0.01, self.size_orb_area, mask=np.array([]), blockSize=3,
                                              useHarrisDetector=0, k=0.04)
            kp2 = []
            for i in range(corners.shape[0]):
                kp_1 = cv2.KeyPoint(corners[i][0][0], corners[i][0][1], self.brief_size)
                kp2.append(kp_1)
            detections_kps = kp2
        else:
            orb = cv2.ORB_create(nfeatures=nkps, edgeThreshold=self.brief_size, patchSize=self.brief_size)
            detections_kps = orb.detect(img2, None)

        if enable_clust:
            clust_kps = self.orb.clust_keypoints(detections_kps)
            detections_kps = clust_kps
        self.detections_kps = detections_kps
        if len(self.tracks) != 0 and img is not None:
            kp2 = []
            if detections_kps is not None:
                detections_arr = np.array([detections_kps[0].pt])
                if len(detections_kps) != 0:
                    for i in range(1, len(detections_kps)):
                        detections_arr = np.concatenate([detections_arr, [detections_kps[i].pt]])
                distance = np.array([])
                for i in range(len(self.tracks)):
                    cur_track = np.tile((self.tracks[i].prediction[0][0], self.tracks[i].prediction[1][0]), (len(detections_kps), 1))
                    if len(distance) == 0:
                        distance = [np.sqrt((detections_arr[:, 0] - cur_track[:, 0])**2 + (detections_arr[:, 1] - cur_track[:, 1])**2)]
                    else:
                        distance = np.concatenate([distance, [np.sqrt((detections_arr[:, 0] - cur_track[:, 0])**2 + (detections_arr[:, 1] - cur_track[:, 1])**2)]])
                euc_distance = distance
                assignment = np.tile(-1, (len(self.tracks),))
                if tracker_hungarian:
                    cost = 0.5 * np.array(distance)
                    row_ind, col_ind = linear_sum_assignment(cost)
                    for i in range(len(row_ind)):
                        assignment[row_ind[i]] = col_ind[i]
                    for i in range(len(assignment)):
                        if assignment[i] != -1:
                            if cost[i][assignment[i]] > self.dist_thresh:
                                assignment[i] = -1
                if tracker_conj_orb_hung:
                    detections_kps, des_kps = orb.compute(img2, detections_kps)
                    distance_Hamming = np.tile(500, (len(self.tracks), len(detections_kps)))
                    for i in range(len(self.tracks)):
                        if min(self.tracks[i].prediction[0][0], self.tracks[i].prediction[1][0],
                                   img2.shape[0] - self.tracks[i].prediction[1][0],
                                   img2.shape[1] - self.tracks[i].prediction[0][0]) < self.brief_size + 1:
                            continue
                        kp_pred = self.tracks[i].key_point
                        [kp_pred], des1 = orb.compute(img2, [kp_pred])
                        if len(self.tracks[i].distance_Hamm) > 20:
                            self.tracks[i].distance_Hamm = []
                        for j in range(len(detections_kps)):
                            distance_Hamming[i][j] = cv2.norm(des_kps[j], des1[0], cv2.NORM_HAMMING)
                    distance = euc_distance *(distance_Hamming)
                    cost = 0.5 * np.array(distance)
                    row_ind, col_ind = linear_sum_assignment(cost)
                    for i in range(len(row_ind)):
                        assignment[row_ind[i]] = col_ind[i]
                    for i in range(len(assignment)):
                        if assignment[i] != -1:
                            if euc_distance[i][assignment[i]] > self.dist_thresh:
                                assignment[i] = -1
                            else:
                                self.tracks[i].distance_Hamm.append(distance_Hamming[i][assignment[i]])
                if tracker_orb:
                    kp_matches = []
                    matches_distance = []
                    euc_distance_list = []
                    for i in range(len(self.tracks)):
                        if min(self.tracks[i].prediction[0][0], self.tracks[i].prediction[1][0],
                                   img.shape[0] - self.tracks[i].prediction[1][0],
                                   img.shape[1] - self.tracks[i].prediction[0][0]) < self.brief_size + 1:
                            kp_matches.append([])
                            matches_distance.append(500)
                            euc_distance_list.append(500)
                            continue
                        kp_pred = self.tracks[i].key_point
                        [kp_id, match_distance] = self.orb.track_kp_selected(img2, nkps, self.size_orb_area,
                                                                                detections_kps, distance[i], kp_pred,
                                                                                self.brief_size)
                        kp_matches.append(detections_kps[kp_id])
                        matches_distance.append(match_distance)
                        euc_distance_list.append(euc_distance[i][kp_id])
                        assignment[i] = kp_id
                    for i in range(len(assignment)):
                        if assignment[i] != -1:

                            if euc_distance_list[i] > self.dist_thresh:
                                assignment[i] = -1
                for i in range(len(assignment)):
                    self.tracks[i].KF.predict(self.tracks[i].prediction)
                    if (assignment[i] != -1):
                        self.tracks[i].invisible_count = 0
                        self.tracks[i].key_point = detections_kps[assignment[i]]
                        if tracker_hungarian or tracker_conj_orb_hung:
                            self.tracks[i].prediction = self.tracks[i].KF.correct(np.array([[detections_kps[assignment[i]].pt[0]],
                                                                                        [detections_kps[assignment[i]].pt[1]]]), 1)

                        if tracker_orb:
                            self.tracks[i].prediction = self.tracks[i].KF.correct(np.array([[kp_matches[i].pt[0]],
                                                            [kp_matches[i].pt[1]]]), 1)
                    else:
                        self.tracks[i].prediction = self.tracks[i].KF.correct(np.array([[0], [0]]), 0)
                        self.tracks[i].invisible_count += 1
                    self.tracks[i].KF.lastResult = self.tracks[i].prediction
                del_tracks = []
                for i in range(len(self.tracks)):
                    if self.tracks[i].invisible_count > self.max_frames_to_skip:
                        del_tracks.append(i)
                if len(del_tracks) > 0:
                    for i in del_tracks:
                        if i < len(self.tracks):
                            del self.tracks[i]
        pass

    def draw_tracks(self, img, en_roi, roi_rect):
        img3 = img
        if img is not None:
            if en_roi:
                self.detections_kps_recount = []
                for i in range(len(self.detections_kps)):
                    self.detections_kps_recount.append(cv2.KeyPoint(self.detections_kps[i].pt[0] + roi_rect[0], self.detections_kps[i].pt[1] + roi_rect[1], self.brief_size))
                self.detections_kps = self.detections_kps_recount
                img3 = cv2.rectangle(img3, (roi_rect[0], roi_rect[1]), (roi_rect[0] + roi_rect[2], roi_rect[1] + roi_rect[3]), (0, 0, 255), 1)
                img3 = cv2.drawKeypoints(img, self.detections_kps, outImage=None, color=(0, 255, 0), flags=0)
                for i in range(len(self.tracks)):
                    if self.tracks[i].invisible_count != 0:
                        img3 = cv2.circle(img3,
                                          (int(self.tracks[i].prediction[0][0]) + roi_rect[0], int(self.tracks[i].prediction[1][0]) + roi_rect[1]),
                                          4, (0, 255, 255), 2)
                    else:
                        img3 = cv2.circle(img3,
                                          (int(self.tracks[i].prediction[0][0]) + roi_rect[0], int(self.tracks[i].prediction[1][0]) + roi_rect[1]),
                                          4, (255, 255, 0), 2)

                    img3 = cv2.putText(img3, str(i), (int(self.tracks[i].prediction[0][0]) + roi_rect[0], int(self.tracks[i].prediction[1][0]) + roi_rect[1]),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                img3 = cv2.drawKeypoints(img, self.detections_kps, outImage=None, color=(0, 255, 0), flags=0)
                for i in range(len(self.tracks)):
                    if self.tracks[i].invisible_count != 0:
                        img3 = cv2.circle(img3,
                                          (int(self.tracks[i].prediction[0][0]), int(self.tracks[i].prediction[1][0])),
                                          4, (0, 255, 255), 2)
                    else:
                        img3 = cv2.circle(img3,
                                          (int(self.tracks[i].prediction[0][0]), int(self.tracks[i].prediction[1][0])),
                                          4, (255, 255, 0), 2)

                    img3 = cv2.putText(img3, str(i), (int(self.tracks[i].prediction[0][0]),
                                                      int(self.tracks[i].prediction[1][0])),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        return img3




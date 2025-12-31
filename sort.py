# SORT: Simple Online and Realtime Tracking
# -----------------------------------------
# Author: Alex Bewley (original SORT)
# Clean Python version for vehicle tracking project

import numpy as np
from filterpy.kalman import KalmanFilter


# -------------------------------
# KalmanBoxTracker
# -------------------------------
class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        """
        bbox: [x1, y1, x2, y2]
        """

        # Create Kalman filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State: x, y, s, r, vx, vy, vs
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.

        # Initialize state
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / (y2 - y1 + 1e-6)

        self.kf.x[:4] = np.array([cx, cy, s, r]).reshape((4, 1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Update state with detected bounding box
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / (y2 - y1 + 1e-6)

        self.kf.update(np.array([cx, cy, s, r]))

    def predict(self):
        """
        Predict next bounding box
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1

        self.history.append(self.get_state())
        return self.history[-1]

    def get_state(self):
        """
        Return predicted bounding box [x1, y1, x2, y2]
        """
        cx, cy, s, r = self.kf.x[:4].reshape((4,))
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])


# --------------------------------------------------
# SORT Tracker wrapper
# --------------------------------------------------
def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)

    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh + 1e-6)

    return o


class Sort:
    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        """
        max_age: frames to wait before deleting track
        min_hits: minimum hits before reporting a track
        iou_threshold: for assignment
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - list of detections [x1,y1,x2,y2,score]
        Returns:
          tracked objects [x1, y1, x2, y2, id]
        """

        self.frame_count += 1

        # Predict new locations for existing tracks
        preds = []
        for t in self.trackers:
            pred = t.predict()
            preds.append(pred)

        preds = np.array(preds)

        # Assign detections to trackers via IOU
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, preds, self.iou_threshold)

        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]][:4])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i][:4])
            self.trackers.append(trk)

        # Remove dead trackers
        output = []
        for t in reversed(self.trackers):
            if t.time_since_update > self.max_age:
                self.trackers.remove(t)
                continue

            if (t.hits >= self.min_hits) or (self.frame_count <= self.min_hits):
                d = t.get_state()
                output.append(np.concatenate((d, [t.id])))

        return np.array(output)


# --------------------------------------------------
# Matching (Hungarian-like simple version)
# --------------------------------------------------
def associate_detections_to_trackers(detections, trackers, iou_threshold):
    if len(trackers) == 0:
        return [], np.arange(len(detections)), []

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    matched_indices = []
    for d in range(len(detections)):
        best_t = np.argmax(iou_matrix[d])
        if iou_matrix[d][best_t] >= iou_threshold:
            matched_indices.append([d, best_t])

    matched_indices = np.array(matched_indices)

    # Unmatched detections
    matched_dets = matched_indices[:, 0] if matched_indices.size > 0 else []
    unmatched_dets = np.setdiff1d(np.arange(len(detections)), matched_dets)

    # Unmatched trackers
    unmatched_trks = []
    matched_trks = matched_indices[:, 1] if matched_indices.size > 0 else []

    for t in range(len(trackers)):
        if t not in matched_trks:
            unmatched_trks.append(t)

    return matched_indices, unmatched_dets, unmatched_trks

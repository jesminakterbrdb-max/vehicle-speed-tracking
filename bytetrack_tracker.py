# ByteTrack Tracker (Python lightweight version)
# ---------------------------------------------------

import numpy as np

class Track:
    def __init__(self, box, track_id):
        self.box = box  # [x1, y1, x2, y2]
        self.id = track_id
        self.lost = 0   # frames lost


class ByteTracker:
    def __init__(self, max_lost=20):
        self.next_id = 1
        self.tracks = []
        self.max_lost = max_lost

    def iou(self, b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])

        union = area1 + area2 - inter + 1e-6
        return inter / union

    def update(self, detections):
        updated_tracks = []

        used_det = set()

        # Assign detections to existing tracks
        for track in self.tracks:
            best_iou = 0
            best_det = None

            for i, det in enumerate(detections):
                if i in used_det:
                    continue

                iou_val = self.iou(track.box, det)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det = i

            # Update track
            if best_iou > 0.2:
                track.box = detections[best_det]
                track.lost = 0
                used_det.add(best_det)
                updated_tracks.append(track)
            else:
                track.lost += 1
                if track.lost <= self.max_lost:
                    updated_tracks.append(track)

        # Create new tracks for unassigned detections
        for i, det in enumerate(detections):
            if i not in used_det:
                new_track = Track(det, self.next_id)
                self.next_id += 1
                updated_tracks.append(new_track)

        self.tracks = updated_tracks
        return updated_tracks

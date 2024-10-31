import math
import numpy as np
from collections import deque

class Tracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep track of how long an object has disappeared
        self.disappeared = {}
        # Track object's movement history
        self.tracks = {}
        # Store object velocities
        self.velocities = {}
        # Keep the count of the IDs
        self.id_count = 0
        # Maximum frames to keep track of disappeared objects
        self.max_disappeared = max_disappeared
        # Maximum distance for matching
        self.max_distance = max_distance

    def calculate_velocity(self, track_history, current_pos):
        """Calculate velocity based on recent positions"""
        if len(track_history) < 2:
            return (0, 0)
        
        prev_pos = track_history[-2]
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        return (dx, dy)

    def predict_position(self, object_id):
        """Predict next position based on velocity"""
        if object_id not in self.velocities:
            return self.center_points[object_id]
        
        current_pos = self.center_points[object_id]
        vx, vy = self.velocities[object_id]
        predicted_x = current_pos[0] + vx
        predicted_y = current_pos[1] + vy
        return (predicted_x, predicted_y)

    def calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection coordinates
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)
        
        # Calculate area of intersection
        interArea = max(0, xB - xA) * max(0, yB - yA)
        
        # Calculate area of both bounding boxes
        box1Area = w1 * h1
        box2Area = w2 * h2
        
        # Calculate IoU
        iou = interArea / float(box1Area + box2Area - interArea)
        return iou

    def update(self, objects_rect):
        objects_bbs_ids = []
        
        # Handle the case when no objects are detected
        if len(objects_rect) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister_object(object_id)
            return objects_bbs_ids

        # Calculate centroids for new detections
        current_centroids = []
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            current_centroids.append((cx, cy, x, y, w, h))

        # If we have no existing objects, register all as new
        if len(self.center_points) == 0:
            for centroid in current_centroids:
                self.register_object(centroid, objects_rect[len(objects_bbs_ids)])
                objects_bbs_ids.append([*objects_rect[len(objects_bbs_ids)], self.id_count - 1])
        else:
            # Get IDs and predicted positions of existing objects
            object_ids = list(self.center_points.keys())
            predicted_positions = [self.predict_position(object_id) for object_id in object_ids]

            # Calculate distance matrix between predicted positions and new detections
            D = np.zeros((len(predicted_positions), len(current_centroids)))
            for i, predicted_pos in enumerate(predicted_positions):
                for j, centroid in enumerate(current_centroids):
                    # Calculate Euclidean distance
                    distance = math.hypot(predicted_pos[0] - centroid[0], 
                                       predicted_pos[1] - centroid[1])
                    
                    # Calculate IoU between bounding boxes
                    iou = self.calculate_iou(objects_rect[j], 
                                          [self.tracks[object_ids[i]][-1][2],
                                           self.tracks[object_ids[i]][-1][3],
                                           self.tracks[object_ids[i]][-1][4],
                                           self.tracks[object_ids[i]][-1][5]])
                    
                    # Combine distance and IoU for matching score
                    D[i, j] = distance * (1 - iou)

            # Use Hungarian algorithm for optimal matching
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if D[row, col] > self.max_distance:
                    continue

                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.center_points[object_id] = (current_centroids[col][0], current_centroids[col][1])
                self.update_tracks(object_id, current_centroids[col])
                self.disappeared[object_id] = 0
                
                objects_bbs_ids.append([*objects_rect[col], object_id])
                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched detections and objects
            unused_rows = set(range(len(predicted_positions))) - used_rows
            unused_cols = set(range(len(current_centroids))) - used_cols

            # Register new objects for unmatched detections
            for col in unused_cols:
                self.register_object(current_centroids[col], objects_rect[col])
                objects_bbs_ids.append([*objects_rect[col], self.id_count - 1])

            # Mark objects as disappeared if not matched
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister_object(object_id)

        return objects_bbs_ids

    def register_object(self, centroid, rect):
        """Register new object"""
        self.center_points[self.id_count] = (centroid[0], centroid[1])
        self.tracks[self.id_count] = deque([(centroid[0], centroid[1], 
                                           rect[0], rect[1], rect[2], rect[3])], maxlen=30)
        self.velocities[self.id_count] = (0, 0)
        self.disappeared[self.id_count] = 0
        self.id_count += 1

    def deregister_object(self, object_id):
        """Deregister disappeared object"""
        del self.center_points[object_id]
        del self.tracks[object_id]
        del self.velocities[object_id]
        del self.disappeared[object_id]

    def update_tracks(self, object_id, centroid):
        """Update tracking history and velocity"""
        self.tracks[object_id].append((centroid[0], centroid[1], 
                                     centroid[2], centroid[3], 
                                     centroid[4], centroid[5]))
        self.velocities[object_id] = self.calculate_velocity(
            [(p[0], p[1]) for p in self.tracks[object_id]], 
            (centroid[0], centroid[1])
        )
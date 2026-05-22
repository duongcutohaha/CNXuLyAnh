from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared=20):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.total_count = 0
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.total_count += 1
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
        else:
            obj_ids = list(self.objects.keys())
            obj_centroids = list(self.objects.values())

            D = np.linalg.norm(
                np.array(obj_centroids)[:, None] - np.array(input_centroids),
                axis=2
            )

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                obj_id = obj_ids[r]
                self.objects[obj_id] = input_centroids[c]
                self.disappeared[obj_id] = 0
                used_rows.add(r)
                used_cols.add(c)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for r in unused_rows:
                obj_id = obj_ids[r]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            for c in unused_cols:
                self.register(input_centroids[c])

        return self.objects
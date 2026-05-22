from __future__ import annotations

import cv2
import numpy as np


def count_objects(mask, original, min_area: int = 700, distance_ratio: float = 0.35):
    working = original.copy()

    sure_background = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_foreground = cv2.threshold(
        distance,
        distance_ratio * distance.max(),
        255,
        0,
    )
    sure_foreground = sure_foreground.astype(np.uint8)

    unknown = cv2.subtract(sure_background, sure_foreground)
    _, markers = cv2.connectedComponents(sure_foreground)
    markers = markers + 1
    markers[unknown == 255] = 0

    watershed_markers = cv2.watershed(working, markers)

    count = 0
    debug_mask = np.zeros(mask.shape, dtype=np.uint8)

    for marker_id in np.unique(watershed_markers):
        if marker_id <= 1:
            continue

        component = np.zeros(mask.shape, dtype=np.uint8)
        component[watershed_markers == marker_id] = 255
        area = cv2.countNonZero(component)
        if area < min_area:
            continue

        count += 1
        debug_mask = cv2.bitwise_or(debug_mask, component)

        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(working, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            working,
            str(count),
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    overlay = working.copy()
    overlay[debug_mask == 255] = (0, 255, 255)
    result = cv2.addWeighted(overlay, 0.25, working, 0.75, 0)

    marker_preview = cv2.normalize(
        watershed_markers.astype(np.float32),
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    return count, result, marker_preview

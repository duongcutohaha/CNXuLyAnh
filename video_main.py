import cv2
import numpy as np
from tracker import CentroidTracker

from utils.preprocess import preprocess_image
from utils.detect import create_fruit_mask

def main():
    cap = cv2.VideoCapture("video.mp4")  # hoặc 0 nếu webcam
    tracker = CentroidTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (600, 400))

        processed = preprocess_image(frame)
        mask = create_fruit_mask(processed)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids = []

        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w//2, y + h//2

                centroids.append((cx, cy))
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        objects = tracker.update(centroids)

        for obj_id, (cx, cy) in objects.items():
            cv2.putText(frame, f"ID {obj_id}", (cx-10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            cv2.circle(frame, (cx, cy), 4, (255,0,0), -1)

        cv2.putText(frame, f"Total: {tracker.total_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Fruit Counter Video", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
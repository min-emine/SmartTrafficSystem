# src/detector.py
from ultralytics import YOLO
import supervision as sv
import cv2

class TrafficDetector:
    def __init__(self, model_path):
        print(f"Model yükleniyor: {model_path}...")
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack() 

    def detect_and_track(self, frame):
        """
        Görüntüyü alır, YOLO ile tarar ve Takip (Tracking) ekleyerek döndürür.
        """
        # Sadece Araçları Tespit Et (2:Car, 3:Motor, 5:Bus, 7:Truck)
        results = self.model(frame, classes=[2, 3, 5, 7], verbose=False)[0]
        
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)
        
        return detections
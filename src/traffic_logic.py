# src/traffic_logic.py
import numpy as np
from sklearn.cluster import KMeans
import supervision as sv  

class TrafficBrain:
    def __init__(self, config):
        self.learning_frames = config['ai']['learning_frames']
        self.n_clusters = config['ai']['cluster_count']
        self.weights = config['weights']
        
        self.frame_count = 0
        self.is_learning = True
        self.vehicle_centroids = [] # Öğrenme verisi
        self.kmeans = None
        self.cluster_centers = []
        
        # Karar çıktıları
        self.current_scores = {}
        self.best_lane = -1

    def update(self, detections):
        """
        Her karede çalışır. Önce öğrenir, sonra yönetir.
        """

        points = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)

        # --- MOD 1: ÖĞRENME ---
        if self.is_learning:
            for pt in points:
                self.vehicle_centroids.append(pt)
            
            self.frame_count += 1
            progress = (self.frame_count / self.learning_frames) * 100
            
            # Süre dolduysa eğitimi bitir
            if self.frame_count >= self.learning_frames:
                self._train_model()
            
            return {"status": "LEARNING", "progress": progress}

        # --- MOD 2: YÖNETME (ACTIVE) ---
        else:
            if len(points) == 0:
                return {"status": "ACTIVE", "scores": {}, "best": -1, "centers": self.cluster_centers}

            # 1. Hangi araç hangi yolda?
            labels = self.kmeans.predict(points)
            
            # 2. Puanları Hesapla
            scores = {i: 0.0 for i in range(self.n_clusters)}
            
            for label, class_id in zip(labels, detections.class_id):
                # YAML'dan gelen ağırlığı kullan, yoksa 1.0 ver
                w = self.weights.get(class_id, 1.0)
                scores[label] += w
            
            self.current_scores = scores
            
            # 3. Karar Ver (En yüksek puanlı yol)
            if max(scores.values()) > 0:
                self.best_lane = max(scores, key=scores.get)
            
            return {
                "status": "ACTIVE",
                "scores": scores,
                "best": self.best_lane,
                "centers": self.cluster_centers
            }

    def _train_model(self):
        print("Eğitim verisi toparlandı, yollar haritalanıyor...")
        if len(self.vehicle_centroids) > self.n_clusters * 2:
            data = np.array(self.vehicle_centroids)
            self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10).fit(data)
            self.cluster_centers = self.kmeans.cluster_centers_
            self.is_learning = False
            print(f"YAPAY ZEKA HAZIR: {self.n_clusters} adet güzergah öğrenildi.")
        else:
            print("Yetersiz veri! Öğrenme süresi 50 kare uzatılıyor.")
            self.learning_frames += 50
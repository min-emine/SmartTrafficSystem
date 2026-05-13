# src/traffic_logic.py
import numpy as np
from sklearn.cluster import KMeans
import supervision as sv  

class TrafficBrain:
    def __init__(self, config):
        self.learning_frames = config['ai']['learning_frames']
        self.n_clusters = config['ai']['cluster_count']
        self.weights = config['weights']
        self.resolution = tuple(config.get('system', {}).get('resolution', [1280, 720]))
        
        self.frame_count = 0
        self.is_learning = True
        self.vehicle_centroids = [] # Öğrenme verisi (normalized raw points backup)
        self.track_histories = {}   # track_id -> list of normalized points
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
        # normalize coordinates to [0,1] using configured resolution for more stable clustering
        try:
            pts_arr = np.array(points, dtype=float)
            w, h = float(self.resolution[0]), float(self.resolution[1])
            if pts_arr.size != 0:
                norm_points = (pts_arr / np.array([w, h]))
            else:
                norm_points = pts_arr
        except Exception:
            norm_points = np.array(points, dtype=float)

        # Debug: show incoming detection info
        try:
            cls_ids = list(detections.class_id)
        except Exception:
            cls_ids = []

        if len(points) > 0:
            sample_pts = points[:6]
            sample_norm = norm_points[:6].tolist() if hasattr(norm_points, 'tolist') else []
        else:
            sample_pts = []

        print(f"[TrafficBrain] frame={self.frame_count} pts={len(points)} sample={sample_pts} sample_norm={sample_norm} classes={cls_ids[:6]}")

        # --- MOD 1: ÖĞRENME ---
        if self.is_learning:
            # attempt to get tracker ids to build per-track histories
            track_ids = None
            for key in ("track_id", "tracker_id", "ids", "id", "trackids", "tracking_id"):
                track_ids = getattr(detections, key, None)
                if track_ids is not None:
                    break

            # store per-point backup
            for pt in norm_points:
                self.vehicle_centroids.append(pt.tolist())

            # store per-track histories when available
            if track_ids is not None:
                try:
                    for tid, pt in zip(track_ids, norm_points):
                        if tid is None:
                            continue
                        tid_key = int(tid)
                        self.track_histories.setdefault(tid_key, []).append(pt.tolist())
                except Exception:
                    pass
            
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
            # Build prediction features matching training shape (2D or 4D)
            try:
                n_features = None
                try:
                    n_features = getattr(self.kmeans, 'n_features_in_', None)
                except Exception:
                    pass
                if n_features is None:
                    try:
                        n_features = self.kmeans.cluster_centers_.shape[1]
                    except Exception:
                        n_features = 2

                if n_features == 2:
                    feat_for_pred = norm_points
                else:
                    # try to get track ids to estimate velocity per detection
                    track_ids = None
                    for key in ("track_id", "tracker_id", "ids", "id", "trackids", "tracking_id"):
                        track_ids = getattr(detections, key, None)
                        if track_ids is not None:
                            break

                    preds = []
                    for idx, p in enumerate(norm_points):
                        vxvy = [0.0, 0.0]
                        try:
                            if track_ids is not None:
                                tid = int(track_ids[idx])
                                hist = self.track_histories.get(tid, None)
                                if hist is not None and len(hist) >= 2:
                                    arr = np.array(hist, dtype=float)
                                    vxvy = (arr[-1] - arr[0]).tolist()
                        except Exception:
                            pass
                        preds.append(np.hstack([p, vxvy]))
                    feat_for_pred = np.array(preds, dtype=float)

                labels = self.kmeans.predict(feat_for_pred)
            except Exception:
                # final fallback: try 2D points
                labels = self.kmeans.predict(np.array(points, dtype=float))

            # Debug: show predicted label distribution
            try:
                from collections import Counter

                label_counts = Counter(labels)
                print(f"[TrafficBrain] Predicted labels counts: {label_counts}")
                print(f"[TrafficBrain] Points sample with labels: {[ (pt.tolist(), lbl) for pt, lbl in zip(norm_points[:8], labels[:8]) ]}")
                print(f"[TrafficBrain] Detections class ids sample: {cls_ids[:8]}")
            except Exception:
                pass
            
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
        # Build per-track feature vectors: [mean_x, mean_y, vx, vy]
        features = []
        for tid, hist in list(self.track_histories.items()):
            if len(hist) < 2:
                continue
            arr = np.array(hist, dtype=float)
            mean_xy = arr.mean(axis=0)
            # velocity: last - first
            v = arr[-1] - arr[0]
            features.append(np.hstack([mean_xy, v]))

        # fallback: if no per-track features, use raw normalized points (as 2D -> pad zeros)
        if len(features) == 0 and len(self.vehicle_centroids) > 0:
            raw = np.array(self.vehicle_centroids, dtype=float)
            for p in raw:
                features.append(np.hstack([p, [0.0, 0.0]]))

        if len(features) >= self.n_clusters:
            data = np.array(features)
            self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, init='k-means++').fit(data)
            self.cluster_centers = self.kmeans.cluster_centers_
            print(f"[TrafficBrain] Trained KMeans centers: {self.cluster_centers}")
            self.is_learning = False
            print(f"YAPAY ZEKA HAZIR: {self.n_clusters} adet güzergah öğrenildi.")
        else:
            print("Yetersiz track-temelli veri! Öğrenme süresi 50 kare uzatılıyor.")
            self.learning_frames += 50
        
# src/visualizer.py
import cv2
import supervision as sv

class Visualizer:
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator(thickness=1)
        
    def draw(self, frame, detections, logic_data):
        # 1. Araç Kutuları
        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        
        status = logic_data["status"]

        # 2. Duruma Göre Çizimler
        if status == "LEARNING":
            progress = logic_data["progress"]
            h, w, _ = frame.shape
            # İlerleme Çubuğu
            cv2.rectangle(frame, (50, h-80), (w-50, h-30), (50, 50, 50), -1)
            bar_width = int(((w-100) * progress) / 100)
            cv2.rectangle(frame, (50, h-80), (50 + bar_width, h-30), (0, 255, 0), -1)
            cv2.putText(frame, f"AI YOLLARI OGRENIYOR... %{int(progress)}", 
                        (w//2 - 150, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        elif status == "ACTIVE":
            centers = logic_data["centers"]
            scores = logic_data["scores"]
            best_lane = logic_data["best"]

            # Yolları ve Işıkları Çiz
            overlay = frame.copy()
            for i, center in enumerate(centers):
                cx, cy = int(center[0]), int(center[1])
                score = scores.get(i, 0)
                
                # Yeşil mi Kırmızı mı?
                is_green = (i == best_lane) and (score > 0)
                color = (0, 255, 0) if is_green else (0, 0, 255)
                
                # Işık Dairesi
                cv2.circle(overlay, (cx, cy), 50, color, -1)
                
                # Yazılar
                cv2.putText(frame, f"YOL-{i+1}", (cx-30, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Puan:{score:.1f}", (cx-35, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Yarı saydamlık uygula
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            
            # Sol Üst Panel
            cv2.rectangle(frame, (10, 10), (400, 80), (0,0,0), -1)
            cv2.putText(frame, "AI TRAFIK KONTROLU", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            if best_lane != -1:
                cv2.putText(frame, f"GECIS: YOL-{best_lane+1}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            else:
                cv2.putText(frame, "TRAFIK YOK", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        return frame
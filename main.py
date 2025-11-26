# main.py
import cv2
import yaml
import streamlink
from src.detector import TrafficDetector
from src.traffic_logic import TrafficBrain
from src.visualizer import Visualizer

def load_config():
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)

def get_stream_url(url):
    try:
        if ".m3u8" in url: return url
        s = streamlink.streams(url)
        return s['best'].url if s else None
    except: return url if ".m3u8" in url else None

def main():
    # 1. Hazırlık
    print("Sistem başlatılıyor...")
    config = load_config()
    
    # Modülleri Başlat (Dependency Injection mantığına yakın)
    detector = TrafficDetector(config['system']['model_path'])
    brain = TrafficBrain(config)
    visualizer = Visualizer()
    
    # Video Akışı
    url = get_stream_url(config['system']['video_url'])
    cap = cv2.VideoCapture(url)
    
    # Ekran Ayarı
    target_w, target_h = config['system']['resolution']
    cv2.namedWindow("Smart Traffic System v2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Smart Traffic System v2", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Yayın koptu, yeniden bağlanılıyor...")
            cap.release()
            cap = cv2.VideoCapture(get_stream_url(config['system']['video_url']))
            continue

        # Görüntüyü Standartlaştır
        frame = cv2.resize(frame, (target_w, target_h))

        # --- AKIŞ (PIPELINE) ---
        # 1. GÖRME (Detect)
        detections = detector.detect_and_track(frame)
        
        # 2. DÜŞÜNME (Logic)
        logic_data = brain.update(detections)
        
        # 3. GÖRSELLEŞTİRME (Draw)
        frame = visualizer.draw(frame, detections, logic_data)

        # 4. GÖSTERME
        cv2.imshow("Smart Traffic System v2", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
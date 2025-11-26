import cv2
import json
import numpy as np
import os
import sys


TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

CONFIG_FILE = "traffic_config.json"
DEFAULT_URL = "https://content.tvkur.com/l/c77i6m384cnrb6mlji4g/master.m3u8"

polygons = [] 
current_polygon = [] 

def load_or_create_config():
    default_data = {"stream_url": DEFAULT_URL, "zones": []}
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'w') as f: json.dump(default_data, f)
        return default_data
    try:
        with open(CONFIG_FILE, 'r') as f: return json.load(f)
    except: return default_data

def mouse_callback(event, x, y, flags, param):
    global current_polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_polygon: current_polygon.pop()

def main():
    global current_polygon, polygons 
    config = load_or_create_config()
    url = config.get("stream_url", DEFAULT_URL)
    
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Video açılamadı!"); return

    cv2.namedWindow("Bolge Cizim Araci", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Bolge Cizim Araci", TARGET_WIDTH, TARGET_HEIGHT)
    cv2.setMouseCallback("Bolge Cizim Araci", mouse_callback)

    print(f"\n--- CIZIM MODU ({TARGET_WIDTH}x{TARGET_HEIGHT}) ---")
    print("Mouse ile yollari cizin. 'N' ile kaydet, 'S' ile cik.")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release(); cap = cv2.VideoCapture(url); continue

       
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

        for poly in polygons:
            pts = np.array(poly["points"], np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            cv2.putText(frame, poly["name"], (pts[0][0], pts[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if len(current_polygon) > 0:
            pts = np.array(current_polygon, np.int32)
            cv2.polylines(frame, [pts], False, (0, 0, 255), 2)
            for pt in current_polygon: cv2.circle(frame, tuple(pt), 4, (0, 0, 255), -1)

        cv2.imshow("Bolge Cizim Araci", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):
            if len(current_polygon) > 2:
                print(">>> TERMINALE ISIM GIRIN <<<")
                try: name = input("Bolge Adi: ")
                except: name = "Bolge"
                polygons.append({"name": name, "points": list(current_polygon)})
                current_polygon = []
            else: print("En az 3 nokta!")
        elif key == ord('s'):
            config["zones"] = polygons
            with open(CONFIG_FILE, 'w') as f: json.dump(config, f, indent=4)
            print("Kaydedildi!"); break
        elif key == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()



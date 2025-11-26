import cv2
import json
import numpy as np
import streamlink
from ultralytics import YOLO
import supervision as sv
import os


TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

CONFIG_FILE = "traffic_config.json"
MODEL_PATH = "yolo11n.pt"

def get_stream(url):
    try:
        if ".m3u8" in url: return url
        s = streamlink.streams(url)
        return s['best'].url if s else None
    except: return url if ".m3u8" in url else None

def main():
    if not os.path.exists(CONFIG_FILE): print("Once cizim yap!"); return
    with open(CONFIG_FILE) as f: config = json.load(f)
    
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(get_stream(config["stream_url"]))
    
    zones, zone_ann = [], []
    colors = sv.ColorPalette.DEFAULT
    for i, z in enumerate(config["zones"]):
        zone = sv.PolygonZone(np.array(z["points"]))
        zones.append(zone)
        zone_ann.append(sv.PolygonZoneAnnotator(zone=zone, color=colors.by_idx(i), thickness=2, text_scale=0.8))

    box_ann = sv.BoxAnnotator(thickness=1)
    tracker = sv.ByteTrack()

    cv2.namedWindow("Trafik", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Trafik", TARGET_WIDTH, TARGET_HEIGHT)

    print("Sistem Basladi...")

    while True:
        ret, frame = cap.read()
        if not ret: 
            cap.release(); cap = cv2.VideoCapture(get_stream(config["stream_url"])); continue


        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

        res = model(frame, classes=[2,3,5,7], verbose=False)[0]
        dets = sv.Detections.from_ultralytics(res)
        dets = tracker.update_with_detections(dets)
        
        frame = box_ann.annotate(scene=frame, detections=dets)

        panel_data = []
        for i, zone in enumerate(zones):
            zone.trigger(detections=dets)
            frame = zone_ann[i].annotate(scene=frame)
            panel_data.append((config["zones"][i]["name"], zone.current_count))

        # Panel
        cv2.rectangle(frame, (10, 10), (350, 50 + len(panel_data)*40), (0,0,0), -1)
        for i, (name, cnt) in enumerate(panel_data):
            col = (0,255,0) if cnt < 5 else ((0,255,255) if cnt < 10 else (0,0,255))
            cv2.putText(frame, f"{name}: {cnt}", (20, 40 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        cv2.imshow("Trafik", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()


import json
import time
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import cv2
import yaml

try:
    import streamlink
except ImportError:
    streamlink = None

from src.detector import TrafficDetector
from src.traffic_logic import TrafficBrain
from src.visualizer import Visualizer


DASHBOARD_DIR = Path("dashboard")
DASHBOARD_ROOT = Path(__file__).resolve().parent
DASHBOARD_PORT = "3001"
STATE_WRITE_INTERVAL_SECONDS = 1.0
DASHBOARD_OUT_DIR = DASHBOARD_DIR / "out"
TRAFFIC_STATE_LOCK = threading.Lock()
TRAFFIC_STATE_JSON = json.dumps({
    "mode": "autonomous",
    "status": "loading",
    "progress": 0,
    "learningFrames": 150,
    "clusterCount": 4,
    "vehicleCount": 0,
    "bestLane": -1,
    "scores": [0, 0, 0, 0],
    "laneSignals": ["red", "red", "red", "red"],
    "videoUrl": "/kayit.mp4",
    "source": "python-runtime",
    "updatedAt": datetime.now(timezone.utc).isoformat(),
}, ensure_ascii=False, indent=2)

ROAD_LABELS = [
    {"name": "Bati", "label": "Bati Yol"},
    {"name": "Yanyol", "label": "Yanyol"},
    {"name": "Kuzey Sag", "label": "Kuzey Sag"},
    {"name": "Kuzey Sol", "label": "Kuzey Sol"},
]


def load_config():
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_stream_url(url):
    try:
        if streamlink is None:
            return url
        if ".m3u8" in url:
            return url
        streams = streamlink.streams(url)
        return streams["best"].url if streams else None
    except Exception:
        return url if ".m3u8" in url else None


def get_dashboard_video_url(source_url):
    if not source_url:
        return "/kayit.mp4"

    if "://" in source_url or source_url.startswith("/"):
        return source_url

    return f"/{Path(source_url).name}"


def lane_signals(best_lane, lane_count):
    signals = ["red"] * lane_count
    if best_lane is not None and best_lane >= 0 and best_lane < lane_count:
        signals[best_lane] = "green"
        if lane_count > 1:
            signals[(best_lane + 1) % lane_count] = "amber"
    return signals


def format_wait_time(score, is_best_lane):
    base_seconds = 25 + int(min(65, round(score * 12)))
    if is_best_lane:
        base_seconds = max(12, base_seconds - 8)
    minutes, seconds = divmod(base_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def build_lane_snapshot(lane_count, scores, best_lane, lane_signals_value, vehicle_count):
    if lane_count <= 0:
        return []

    total_score = sum(scores) or 1.0
    snapshots = []
    for index in range(lane_count):
        score = float(scores[index] if index < len(scores) else 0.0)
        share = score / total_score
        vehicle_estimate = max(0, int(round(vehicle_count * share)))
        is_best_lane = index == best_lane
        road = ROAD_LABELS[index] if index < len(ROAD_LABELS) else {"name": f"Route {index + 1}", "label": f"Route {index + 1}"}
        snapshots.append({
            "id": index,
            "name": road["name"],
            "direction": road["label"],
            "signal": lane_signals_value[index] if index < len(lane_signals_value) else "red",
            "vehicles": vehicle_estimate,
            "priorityScore": round(score, 2),
            "occupancy": min(100, int(round(share * 100))),
            "avgWait": format_wait_time(score, is_best_lane),
            "trend": "+0%" if score == 0 else ("+" if is_best_lane else "") + f"{int(round(share * 100))}%",
            "emergency": bool(is_best_lane and vehicle_count > 0),
        })

    return snapshots


def build_zone_snapshot(config, lane_count, scores):
    lane_width = int(config["system"].get("resolution", [1280, 720])[0])
    lane_height = int(config["system"].get("resolution", [1280, 720])[1])
    total_score = sum(scores) or 1.0
    zones = []
    for index in range(lane_count):
        share = float(scores[index] if index < len(scores) else 0.0) / total_score
        x = int((index + 1) * lane_width / (lane_count + 1))
        y = int(lane_height * (0.28 + (index % 2) * 0.22))
        road = ROAD_LABELS[index] if index < len(ROAD_LABELS) else {"name": f"Zone {index + 1}", "label": f"Route {index + 1}"}
        zones.append({
            "id": index,
            "name": road["name"],
            "label": road["label"],
            "count": max(0, int(round(share * 10))),
            "center": [x, y],
        })

    return zones


def build_event_snapshot(status, best_lane, vehicle_count, progress):
    road_name = ROAD_LABELS[best_lane]["label"] if 0 <= best_lane < len(ROAD_LABELS) else f"Lane {best_lane + 1}"
    if status == "LEARNING":
        return [
            {
                "time": datetime.now(timezone.utc).strftime("%H:%M"),
                "title": "Learning in progress",
                "detail": f"{progress:.1f}% of the road map has been learned.",
            },
            {
                "time": datetime.now(timezone.utc).strftime("%H:%M"),
                "title": "Video feed attached",
                "detail": "Local kayit.mp4 is driving both the UI and the detector.",
            },
        ]

    return [
        {
            "time": datetime.now(timezone.utc).strftime("%H:%M"),
            "title": f"{road_name} receiving green",
            "detail": f"{vehicle_count} vehicles observed in the active route.",
        },
        {
            "time": datetime.now(timezone.utc).strftime("%H:%M"),
            "title": "Routes synced",
            "detail": "Scores, signals, and zone summaries are being refreshed live.",
        },
    ]


def write_dashboard_state(config, logic_data, vehicle_count):
    lane_count = int(config["ai"]["cluster_count"])
    scores_dict = logic_data.get("scores", {})
    scores = [float(scores_dict.get(i, 0.0)) for i in range(lane_count)]
    best_lane = int(logic_data.get("best", -1))
    video_url = get_dashboard_video_url(config.get("system", {}).get("video_url", ""))
    lane_signals_value = lane_signals(best_lane, lane_count)
    lane_snapshots = build_lane_snapshot(lane_count, scores, best_lane, lane_signals_value, vehicle_count)
    zone_snapshots = build_zone_snapshot(config, lane_count, scores)
    event_snapshots = build_event_snapshot(logic_data["status"], best_lane, int(vehicle_count), float(logic_data.get("progress", 0.0)))

    if logic_data["status"] == "LEARNING":
        payload = {
            "mode": "autonomous",
            "status": "learning",
            "progress": float(logic_data.get("progress", 0.0)),
            "learningFrames": int(config["ai"]["learning_frames"]),
            "clusterCount": lane_count,
            "vehicleCount": int(vehicle_count),
            "bestLane": -1,
            "scores": [0.0] * lane_count,
            "laneSignals": ["red"] * lane_count,
            "lanes": lane_snapshots,
            "zones": zone_snapshots,
            "events": event_snapshots,
            "videoUrl": video_url,
            "source": "python-runtime",
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }
    else:
        payload = {
            "mode": "autonomous",
            "status": "active",
            "progress": 100.0,
            "learningFrames": int(config["ai"]["learning_frames"]),
            "clusterCount": lane_count,
            "vehicleCount": int(vehicle_count),
            "bestLane": best_lane,
            "scores": scores,
            "laneSignals": lane_signals_value,
            "lanes": lane_snapshots,
            "zones": zone_snapshots,
            "events": event_snapshots,
            "videoUrl": video_url,
            "source": "python-runtime",
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }

    global TRAFFIC_STATE_JSON
    with TRAFFIC_STATE_LOCK:
        TRAFFIC_STATE_JSON = json.dumps(payload, ensure_ascii=False, indent=2)


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.rstrip("/") == "/traffic-state.json":
            with TRAFFIC_STATE_LOCK:
                data = TRAFFIC_STATE_JSON.encode("utf-8")

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path.rstrip("/") == "/kayit.mp4":
            video_path = DASHBOARD_ROOT / "kayit.mp4"
            if not video_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "kayit.mp4 not found")
                return

            data = video_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        return super().do_GET()


def start_dashboard():
    if not DASHBOARD_DIR.exists():
        return None

    if not DASHBOARD_OUT_DIR.exists():
        print("Dashboard build çıktısı yok. Önce dashboard klasöründe `npm run build` çalıştırın.")
        return None

    port = int(DASHBOARD_PORT)
    while port < 3020:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                break
        port += 1

    if port >= 3020:
        print("Boş dashboard portu bulunamadı (3001-3019).")
        return None

    try:
        def serve():
            handler = lambda *args, **kwargs: DashboardRequestHandler(*args, directory=str(DASHBOARD_OUT_DIR), **kwargs)
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            print(f"Serving dashboard from {DASHBOARD_OUT_DIR} on port {port}")
            server.serve_forever()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        return thread, port
    except Exception as exc:
        print(f"Dashboard otomatik başlatılamadı: {exc}")
        print("Dashboard için ayrı terminalde: cd dashboard; npm run build; python main.py")
        return None


def main():
    print("Sistem başlatılıyor...")
    config = load_config()

    detector = TrafficDetector(config["system"]["model_path"])
    brain = TrafficBrain(config)
    visualizer = Visualizer()

    dashboard_handle = start_dashboard()
    dashboard_process = None
    if dashboard_handle:
        dashboard_process, dashboard_port = dashboard_handle
        print(f"Web dashboard başlatıldı: http://localhost:{dashboard_port}")

    url = get_stream_url(config["system"]["video_url"])
    cap = cv2.VideoCapture(url)
    target_w, target_h = config["system"]["resolution"]

    preview_window = False
    last_state_write = 0.0
    if preview_window:
        cv2.namedWindow("Smart Traffic System v2", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart Traffic System v2", 1280, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Yayın koptu, yeniden bağlanılıyor...")
                cap.release()
                cap = cv2.VideoCapture(get_stream_url(config["system"]["video_url"]))
                continue

            frame = cv2.resize(frame, (target_w, target_h))
            detections = detector.detect_and_track(frame)
            logic_data = brain.update(detections)

            now = time.monotonic()
            if now - last_state_write >= STATE_WRITE_INTERVAL_SECONDS:
                write_dashboard_state(config, logic_data, len(detections))
                last_state_write = now

            if preview_window:
                frame = visualizer.draw(frame, detections, logic_data)
                cv2.imshow("Smart Traffic System v2", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nSistem durduruldu.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

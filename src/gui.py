# src/gui.py
import tkinter as tk
from tkinter import scrolledtext, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import yaml
from src.detector import TrafficDetector
from src.traffic_logic import TrafficBrain
from src.visualizer import Visualizer
from src.chatbot import TrafficChatBot

class TrafficSystemGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚦 Smart Traffic System v2 + Chatbot")
        self.root.geometry("1800x950")
        self.root.configure(bg="#1e1e1e")
        self.root.minsize(1400, 700)
        
        # Bayraklar
        self.running = False
        self.cap = None
        self.detector = None
        self.brain = None
        self.visualizer = None
        self.chatbot = None
        
        # Ana layout
        self.create_layout()
        
    def create_layout(self):
        """Ana arayüz tasarımı"""
        # Üst başlık
        header = tk.Frame(self.root, bg="#2d2d2d", height=60)
        header.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        header.pack_propagate(False)
        
        title_label = tk.Label(
            header, 
            text="🚦 Smart Traffic System + 🤖 AI Chatbot",
            font=("Arial", 16, "bold"),
            bg="#2d2d2d",
            fg="#00FF00"
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        status_label = tk.Label(
            header,
            text="Status: Hazır",
            font=("Arial", 11),
            bg="#2d2d2d",
            fg="#FFA500"
        )
        status_label.pack(side=tk.RIGHT, padx=20, pady=10)
        self.status_label = status_label
        
        # Ana içerik - 2 sütun (video sol, chat sağ)
        content = tk.Frame(self.root, bg="#1e1e1e")
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        
        # SOL TARAF - Video Panel
        left_panel = tk.Frame(content, bg="#2d2d2d", relief=tk.RAISED, bd=2)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left_panel.grid_propagate(False)
        
        video_title = tk.Label(
            left_panel,
            text="📹 Trafik Video Analizi",
            font=("Arial", 12, "bold"),
            bg="#2d2d2d",
            fg="#00FF00"
        )
        video_title.pack(pady=8)
        
        # Video alanı (FIXED SIZE)
        self.video_label = tk.Label(
            left_panel,
            bg="#000000",
            width=100,
            height=30
        )
        self.video_label.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        
        # Kontrol butonları
        button_frame = tk.Frame(left_panel, bg="#2d2d2d")
        button_frame.pack(pady=8, fill=tk.X, padx=5)
        
        self.start_btn = tk.Button(
            button_frame,
            text="▶️ BAŞLAT",
            command=self.start_traffic_system,
            bg="#00AA00",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=12,
            pady=6
        )
        self.start_btn.pack(side=tk.LEFT, padx=3)
        
        self.stop_btn = tk.Button(
            button_frame,
            text="⏹️ DURDUR",
            command=self.stop_traffic_system,
            bg="#AA0000",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=12,
            pady=6,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=3)
        
        # SAĞ TARAF - Chatbot Panel
        right_panel = tk.Frame(content, bg="#2d2d2d", relief=tk.RAISED, bd=2)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_panel.grid_propagate(False)
        
        chat_title = tk.Label(
            right_panel,
            text="🤖 AI Asistan",
            font=("Arial", 12, "bold"),
            bg="#2d2d2d",
            fg="#00FFFF"
        )
        chat_title.pack(pady=8)
        
        # Chat geçmişi
        self.chat_display = scrolledtext.ScrolledText(
            right_panel,
            bg="#1a1a1a",
            fg="#FFFFFF",
            font=("Arial", 9),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.chat_display.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        
        # Giriş alanı
        input_frame = tk.Frame(right_panel, bg="#2d2d2d")
        input_frame.pack(pady=5, padx=5, fill=tk.X)
        
        self.chat_input = tk.Entry(
            input_frame,
            font=("Arial", 9),
            bg="#333333",
            fg="#FFFFFF",
            insertbackground="white"
        )
        self.chat_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))
        self.chat_input.bind("<Return>", lambda e: self.send_message())
        
        send_btn = tk.Button(
            input_frame,
            text="✉️ GÖNDER",
            command=self.send_message,
            bg="#0066FF",
            fg="white",
            font=("Arial", 9, "bold"),
            padx=8
        )
        send_btn.pack(side=tk.RIGHT)
        
        # Chatbot başlatma
        self.init_chatbot()
        
    def init_chatbot(self):
        """Chatbot'u başlat"""
        try:
            self.chatbot = TrafficChatBot()
            self.append_chat("Bot", "✅ Hoş geldiniz! Trafik hakkında sorularınızı sorabilirsiniz.")
            self.status_label.config(text="Status: Chatbot Hazır ✓")
        except Exception as e:
            messagebox.showerror("Chatbot Hatası", f"Chatbot başlatılamadı: {str(e)}")
            self.append_chat("Bot", f"❌ Hata: {str(e)}")
    
    def start_traffic_system(self):
        """Trafik sistemi başlat"""
        if self.running:
            return
        
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Trafik Sistemi Çalışıyor 🔴", fg="#FF0000")
        
        # Sistem başlatma thread'i
        thread = threading.Thread(target=self.run_traffic_system, daemon=True)
        thread.start()
    
    def run_traffic_system(self):
        """Trafik sistemi döngüsü"""
        try:
            config = self.load_config()
            
            self.detector = TrafficDetector(config['system']['model_path'])
            self.brain = TrafficBrain(config)
            self.visualizer = Visualizer()
            
            # get_stream_url fonksiyonunu inline olarak yazıyoruz
            def get_stream_url(url):
                try:
                    if ".m3u8" in url: return url
                    import streamlink
                    s = streamlink.streams(url)
                    return s['best'].url if s else None
                except: return url if ".m3u8" in url else None
            
            url = get_stream_url(config['system']['video_url'])
            self.cap = cv2.VideoCapture(url)
            
            target_w, target_h = config['system']['resolution']
            
            frame_count = 0
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    self.status_label.config(text="Status: Bağlantı koptu, yeniden deneniyor...", fg="#FF8800")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(get_stream_url(config['system']['video_url']))
                    continue
                
                frame = cv2.resize(frame, (target_w, target_h))
                
                detections = self.detector.detect_and_track(frame)
                logic_data = self.brain.update(detections)
                frame = self.visualizer.draw(frame, detections, logic_data)
                
                # OpenCV BGR -> RGB dönüş
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Tkinter'da göster
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image=image)
                
                self.video_label.config(image=photo)
                self.video_label.image = photo
                
                self.root.update()
                frame_count += 1
                
                if frame_count % 30 == 0:
                    self.status_label.config(text=f"Status: Çalışıyor ({frame_count} kare)", fg="#00AA00")
        
        except Exception as e:
            messagebox.showerror("Hata", f"Trafik sistemi hatası: {str(e)}")
            self.status_label.config(text="Status: Hata!", fg="#FF0000")
        finally:
            self.stop_traffic_system()
    
    def stop_traffic_system(self):
        """Trafik sistemi durdur"""
        self.running = False
        if self.cap:
            self.cap.release()
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Durduruldu", fg="#FFA500")
    
    def load_config(self):
        """YAML config yükle"""
        with open("config/settings.yaml", "r") as f:
            return yaml.safe_load(f)
    
    def send_message(self):
        """Mesaj gönder"""
        message = self.chat_input.get().strip()
        if not message:
            return
        
        # Kullanıcı mesajı göster
        self.append_chat("Siz", message)
        self.chat_input.delete(0, tk.END)
        
        # Bot yanıt thread'de
        thread = threading.Thread(target=self.get_bot_response, args=(message,), daemon=True)
        thread.start()
    
    def get_bot_response(self, message):
        """Bot yanıtı al"""
        try:
            response = self.chatbot.chat_with_user(message)
            self.append_chat("Bot", response)
        except Exception as e:
            self.append_chat("Bot", f"❌ Hata: {str(e)}")
    
    def append_chat(self, sender, message):
        """Chat ekranına mesaj ekle"""
        self.chat_display.config(state=tk.NORMAL)
        
        if sender == "Siz":
            self.chat_display.insert(tk.END, f"\n👤 {sender}:\n", "user")
            self.chat_display.insert(tk.END, f"{message}\n", "user_msg")
        else:
            self.chat_display.insert(tk.END, f"\n🤖 {sender}:\n", "bot")
            self.chat_display.insert(tk.END, f"{message}\n", "bot_msg")
        
        # Renkler
        self.chat_display.tag_config("user", foreground="#00FF00", font=("Arial", 10, "bold"))
        self.chat_display.tag_config("user_msg", foreground="#FFFFFF")
        self.chat_display.tag_config("bot", foreground="#00FFFF", font=("Arial", 10, "bold"))
        self.chat_display.tag_config("bot_msg", foreground="#FFFFFF")
        
        # Aşağı kaydır
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)


def start_gui():
    """GUI başlat"""
    root = tk.Tk()
    app = TrafficSystemGUI(root)
    root.mainloop()


if __name__ == "__main__":
    start_gui()

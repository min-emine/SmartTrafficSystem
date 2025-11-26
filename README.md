# 🚦 AI-Driven Smart Traffic Management System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-green?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-red?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-K_Means-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

**Geleneksel trafik ışıklarını unutun.** Bu proje, şehir içi trafik yönetimini yapay zeka ile otonom hale getiren, görüntü işleme tabanlı akıllı bir kavşak yönetim sistemidir.

Sistem, **Manuel (Çizim)** ve **Otonom (AI)** olmak üzere iki farklı modda çalışabilir. Otonom modda, hiçbir insan müdahalesi olmadan yolları öğrenir, araç yoğunluğunu ve acil durum önceliklerini (Ambulans, İtfaiye vb.) analiz ederek trafik ışıklarını dinamik olarak yönetir.

---

## 🚀 Temel Özellikler

### 🧠 1. Otonom Mod (AI - Self Learning)
* **Otomatik Yol Haritalama:** `Scikit-Learn K-Means Clustering` algoritması ile araçların hareket rotalarını izler ve şeritleri kendi kendine öğrenir.
* **Öncelik Tabanlı Karar:** Sadece araç sayısına bakmaz; araç tipine göre (Ambulans > Otobüs > Otomobil) ağırlıklı puanlama yapar.
* **Dinamik Işık Yönetimi:** En yüksek öncelik puanına sahip yola otomatik olarak geçiş hakkı (Yeşil Işık) verir.

### ✍️ 2. Manuel Mod (Legacy)
* **Bölge Çizim Aracı:** `zone_creator.py` aracı ile kullanıcı videoda istediği bölgeleri (ROI) mouse ile çizer.
* **Bölgesel Sayım:** Belirlenen poligon alanlarına giren araçlar sayılır ve yoğunluk eşiğine göre uyarı verilir.

---


## 📂 Proje Mimarisi

Proje, genişletilebilir ve modüler bir yapıda tasarlanmıştır.



```text
SmartTrafficSystem/
│
├── config/
│   └── settings.yaml       # Otonom sistemin ayarları (URL, Ağırlıklar, Süreler)
│
├── src/                    # Çekirdek Modüller
│   ├── detector.py         # YOLOv11 Model Yönetimi ve Nesne Takibi (ByteTrack)
│   ├── traffic_logic.py    # Yapay Zeka (Beyin): K-Means ve Karar Algoritması
│   └── visualizer.py       # Görselleştirme ve UI Katmanı
│
├── main.py                 # 🟢 OTONOM SİSTEM (Ana Çalıştırma Dosyası)
├── main_manuel.py          # 🟡 MANUEL SİSTEM (Eski versiyon - Çizim tabanlı)
├── zone_creator.py         # 🟡 Manuel sistem için bölge çizim aracı
├── traffic_config.json     # Manuel sistemin koordinat kayıt dosyası
│
├── yolo11n.pt              # YOLOv11 Nano Ağırlık Dosyası
└── requirements.txt        # Gerekli Kütüphaneler



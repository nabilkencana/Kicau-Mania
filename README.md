# 🐱 CAT POSE DETECTOR v5

Aplikasi interaktif pose detection menggunakan webcam untuk mendeteksi gerakan tangan spesifik dan memicu pemutaran video.

## 📋 Daftar Isi
- [Fitur](#fitur)
- [Persyaratan](#persyaratan)
- [Instalasi](#instalasi)
- [Cara Menggunakan](#cara-menggunakan)
- [Konfigurasi](#konfigurasi)
- [Troubleshooting](#troubleshooting)

---

## ✨ Fitur

- **Real-time Hand & Face Detection** menggunakan MediaPipe
- **Pose Recognition**: Deteksi 2 pose sekaligus:
  - ✋ **Tangan KIRI**: Menutup/dekatkan telapak ke hidung
  - 👋 **Tangan KANAN**: Gerakan wave kiri-kanan (minimal 2x ayunan)
- **Video Trigger**: Tahan pose 1 detik → video diputar otomatis
- **Threading**: Kamera tetap aktif saat video diputar
- **Real-time UI**: Progress bar, status indicators, dan visual feedback
- **Webcam 1280x720**: Resolusi tinggi untuk akurasi maksimal

---

## 🔧 Persyaratan

- **Python**: 3.9+
- **Webcam**: Aktif dan terpasang
- **File Video**: `cat_video.mp4` di folder yang sama dengan script

### Library Dependencies
```
opencv-contrib-python>=4.8.0
mediapipe==0.10.14
tensorflow==2.16.1
ml_dtypes==0.3.2
numpy>=1.26.0
```

---

## 📦 Instalasi

### 1. Clone/Download Project
```bash
cd d:\Projek\Kicau Maniaaa
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Atau install manual:
```bash
pip install opencv-contrib-python mediapipe==0.10.14 tensorflow==2.16.1 ml_dtypes==0.3.2 numpy
```

### 3. Verifikasi Instalasi
```bash
python -c "import mediapipe; import cv2; import tensorflow; print('✓ Semua library OK')"
```

---

## ▶️ Cara Menggunakan

### Basic Usage
```bash
python cat_pose_detector.py
```

### Petunjuk dalam Aplikasi

```
==========================================================
  CAT POSE DETECTOR  v5
==========================================================
  Video  : [path-to-cat_video.mp4]
  Tahan  : 1.0 detik

  POSE:
   Tangan KIRI  → tutup hidung (dekatkan telapak ke hidung)
   Tangan KANAN → wave kiri-kanan minimal 2x

  Tekan Q untuk keluar.
==========================================================
```

### Flow Kerja

1. **Aplikasi dimulai** → Kamera menyala menampilkan feed real-time
2. **Posisikan diri**:
   - Tangan kiri: dekatkan telapak ke hidung (tutup)
   - Tangan kanan: lakukan gerakan wave (goyang kiri-kanan)
3. **Tahan pose** selama 1 detik
   - Progress bar akan muncul menunjukkan durasi
   - Saat mencapai 100%, video mulai diputar
4. **Video diputar** di window terpisah (kamera tetap aktif)
5. **Cooldown 1.5 detik** setelah video selesai sebelum bisa main lagi

### Kontrol

| Tombol | Fungsi |
|--------|--------|
| **Q** | Keluar dari aplikasi |
| **Esc** | Keluar dari aplikasi |

---

## ⚙️ Konfigurasi

Edit nilai-nilai ini di awal file `cat_pose_detector.py`:

```python
# Durasi tahan pose sebelum video diputar (detik)
HOLD_DURATION = 1.0

# Jarak max tangan kiri ke hidung (relatif terhadap lebar frame)
# Range: 0.05 (ketat) - 0.20 (longgar)
NOSE_COVER_DIST = 0.13

# Deteksi wave
WAVE_FRAMES = 24        # Jumlah frame untuk mendeteksi wave
WAVE_MIN_SWING = 0.07   # Minimum total ayunan (0-1)
WAVE_MIN_TURNS = 2      # Minimum perubahan arah (ayunan kiri-kanan)

# Webcam settings
# WIDTH: 1280, HEIGHT: 720

# Confidence thresholds
Hands:
  - min_detection_confidence: 0.75
  - min_tracking_confidence: 0.65
  
FaceMesh:
  - min_detection_confidence: 0.7
  - min_tracking_confidence: 0.6
```

### Tips Konfigurasi

- **Pose terlalu sulit?** → Naikkan `NOSE_COVER_DIST` ke 0.15-0.18
- **Wave tidak terdeteksi?** → Turunkan `WAVE_MIN_SWING` ke 0.05
- **Terlalu sensitif?** → Naikkan `WAVE_FRAMES` ke 32-40

---

## 📊 UI Indicators

### Status Badges
```
Kiri  (tutup hidung):      OK  atau  --
Kanan (wave kiri-kanan):   OK  atau  --
```

- 🟢 **OK** = Pose terdeteksi dengan benar
- 🔴 **--** = Pose belum sesuai atau tidak terdeteksi

### Progress Bar
```
[████████░░░░░░░░░░░░░░] 45%
```
- Muncul saat KEDUA pose benar
- Mencapai 100% → video diputar

### Pesan Status

| Pesan | Arti |
|-------|------|
| `Pose: tangan KIRI tutup hidung + tangan KANAN wave` | Pose belum benar |
| `Tangan kiri OK \| Kanan: wave kiri-kanan` | Kiri OK, tunggu kanan |
| `Tangan kanan OK \| Kiri: tutup hidung` | Kanan OK, tunggu kiri |
| `Bagus! Tahan terus... 67%` | Kedua pose benar, tahan terus |
| `Memutar video...` | Video sedang diputar |
| `Ulangi pose untuk main lagi!` | Cooldown - tunggu sebelum retry |

---

## 🐛 Troubleshooting

### Error: `ModuleNotFoundError: No module named 'ml_dtypes'`

**Solusi:**
```bash
pip install --force-reinstall --no-cache-dir ml_dtypes==0.3.2
```

### Error: `AttributeError: module 'ml_dtypes' has no attribute 'float8_e4m3b11'`

**Solusi:**
Kombinasi versi library tidak compatible. Reinstall dengan versi tepat:
```bash
pip install ml_dtypes==0.3.2 tensorflow==2.16.1 mediapipe==0.10.14 --force-reinstall --no-cache-dir
```

### Webcam tidak terdeteksi

**Cek:**
```python
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✓ Webcam OK")
else:
    print("✗ Webcam tidak terdeteksi")
```

**Solusi:**
- Pastikan webcam terpasang dan aktif di Device Manager
- Coba ganti `cv2.VideoCapture(0)` → `cv2.VideoCapture(1)` (jika ada 2 webcam)
- Restart aplikasi

### Pose tidak terdeteksi dengan baik

**Tips:**
1. Pastikan pencahayaan cukup terang
2. Posisi webcam setinggi wajah Anda
3. Jangan terlalu dekat/jauh dari kamera (1-1.5 meter ideal)
4. Tangan harus terlihat jelas di frame
5. Background tidak terlalu kompleks atau gelap

### Video tidak diputar

**Cek:**
1. File `cat_video.mp4` ada di folder yang sama?
   ```bash
   dir cat_video.mp4
   ```
2. Format video valid? (MP4, WebM, AVI)
3. Coba jalankan manual:
   ```bash
   python -c "import cv2; cap=cv2.VideoCapture('cat_video.mp4'); print('OK' if cap.isOpened() else 'FAIL')"
   ```

### Performance lambat

**Optimasi:**
- Kurangi resolusi webcam (800x600 atau 640x480)
- Tutup aplikasi lain yang berat
- Gunakan GPU jika tersedia (TensorFlow akan auto-detect)

---

## 📁 Struktur File

```
Kicau Maniaaa/
├── cat_pose_detector.py       # Main script
├── cat_video.mp4              # Video yang diputar saat pose benar
├── README.md                  # File ini
└── requirements.txt           # Dependencies list (optional)
```

---

## 🎯 Cara Kerja Teknis

### MediaPipe Solutions
- **Hand Landmarks**: 21 key points per tangan untuk deteksi pose
- **Face Mesh**: 468 landmarks di wajah, kami gunakan landmark #4 (ujung hidung)

### Wave Detection Algorithm
1. Buffer posisi wrist (X-axis) dengan ukuran `WAVE_FRAMES`
2. Hitung perubahan arah (direction changes) = "turns"
3. Hitung total swing (max - min dalam buffer)
4. Wave terdeteksi jika: `turns >= WAVE_MIN_TURNS` AND `swing >= WAVE_MIN_SWING`

### Threading
- Thread utama: Capture & display webcam, detect poses
- Thread video: Putar video di window terpisah (tidak blocking main thread)

---

## 📝 License

Project ini bebas digunakan untuk keperluan pribadi dan pendidikan.

---

## 💡 Tips & Trik

### Pose yang Baik
✅ Tangan kiri:
- Telapak tangan menghadap wajah
- Jari terbuka (tidak genggam)
- Dekat ke hidung (~10cm)

✅ Tangan kanan:
- Goyang konsisten kiri-kanan
- Gerakan cukup cepat (bukan gerakan super slow)
- 2 ayunan penuh sebelum tahan

### Debug Mode
Uncomment baris ini untuk melihat detail pose:
```python
# Tambahkan saat test:
print(f"Left OK: {left_ok}, Right OK: {right_ok}")
print(f"Nose distance: {dist}")
print(f"Wave turns: {turns}, swing: {swing}")
```

---

## 🤝 Support

Jika ada bug atau error:
1. Check error message di terminal
2. Cek requirements.txt dan install ulang dependencies
3. Coba reconnect webcam
4. Restart aplikasi

---

**Created with ❤️ for cat pose detection**

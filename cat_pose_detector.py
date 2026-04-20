"""
╔══════════════════════════════════════════════════════╗
║         CAT POSE DETECTOR  v5                        ║
║                                                      ║
║  POSE:                                               ║
║   • Tangan KIRI  → menutup hidung                    ║
║   • Tangan KANAN → gerak kiri-kanan (wave)           ║
║                                                      ║
║  Tahan 1 detik → video diputar (kamera tetap jalan!) ║
╚══════════════════════════════════════════════════════╝
"""

# ── FIX: stub tensorflow agar tidak konflik ml_dtypes ────────────────────────
import sys, types as _types

class _FD:
    @staticmethod
    def do_not_generate_docs(fn):        return fn
    @staticmethod
    def do_not_doc_in_subclasses(fn):    return fn
    @staticmethod
    def do_not_doc_inheritable(fn):      return fn

_fd = _FD()
_t3 = _types.ModuleType("tensorflow.tools.docs");  _t3.doc_controls = _fd
_t2 = _types.ModuleType("tensorflow.tools");        _t2.docs = _t3
_t1 = _types.ModuleType("tensorflow");              _t1.tools = _t2
for _k, _v in [("tensorflow", _t1), ("tensorflow.tools", _t2),
               ("tensorflow.tools.docs", _t3),
               ("tensorflow.tools.docs.doc_controls", _fd)]:
    sys.modules[_k] = _v
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import threading
from collections import deque

# ──────────────────────────────────────────────────────
#  KONFIGURASI
# ──────────────────────────────────────────────────────

VIDEO_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cat_video.mp4")
HOLD_DURATION = 1.0   # detik pose harus ditahan sebelum video diputar

# Jarak max tangan kiri ke hidung (dalam satuan jarak antar-bahu wajah)
NOSE_COVER_DIST = 0.13  # relatif thd lebar frame — lebih ketat = lebih akurat

# Wave detector
WAVE_FRAMES    = 24    # window frame
WAVE_MIN_SWING = 0.07  # min total ayunan X (0–1)
WAVE_MIN_TURNS = 2     # min balik arah

# ──────────────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ──────────────────────────────────────────────────────

_hands_sol     = mp.solutions.hands
_face_mesh_sol = mp.solutions.face_mesh

# Warna PUTIH untuk semua drawing
_WHITE      = (255, 255, 255)
_WHITE_SPEC = mp.solutions.drawing_styles.DrawingSpec(color=_WHITE, thickness=1, circle_radius=2)

_HAND_LM_STYLE   = mp.solutions.drawing_utils.DrawingSpec(color=_WHITE, thickness=1, circle_radius=3)
_HAND_CONN_STYLE = mp.solutions.drawing_utils.DrawingSpec(color=_WHITE, thickness=1)


# ──────────────────────────────────────────────────────
#  VIDEO PLAYER (THREAD TERPISAH)
# ──────────────────────────────────────────────────────

class VideoPlayer(threading.Thread):
    """
    Memutar video di thread terpisah sehingga kamera tetap berjalan.
    """
    WIN = "  Cat Video  "

    def __init__(self, path):
        super().__init__(daemon=True)
        self.path    = path
        self.running = True
        self._done   = threading.Event()

    def run(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            print(f"[ERROR] Tidak bisa membuka video: {self.path}")
            self._done.set()
            return

        fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = max(1, int(1000 / fps))

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(self.WIN, frame)
            if cv2.waitKey(delay) & 0xFF in (ord('q'), ord('Q'), 27):
                break

        cap.release()
        cv2.destroyWindow(self.WIN)
        self._done.set()

    def stop(self):
        self.running = False

    def is_done(self):
        return self._done.is_set()


# ──────────────────────────────────────────────────────
#  WAVE DETECTOR
# ──────────────────────────────────────────────────────

class WaveDetector:
    def __init__(self):
        self.buf = deque(maxlen=WAVE_FRAMES)

    def update(self, x: float):
        self.buf.append(x)

    def reset(self):
        self.buf.clear()

    def is_waving(self) -> bool:
        if len(self.buf) < WAVE_FRAMES // 2:
            return False
        xs       = list(self.buf)
        turns    = 0
        last_dir = 0
        for i in range(1, len(xs)):
            dx = xs[i] - xs[i - 1]
            if abs(dx) < 0.004:
                continue
            d = 1 if dx > 0 else -1
            if last_dir and d != last_dir:
                turns += 1
            last_dir = d
        swing = max(xs) - min(xs)
        return turns >= WAVE_MIN_TURNS and swing >= WAVE_MIN_SWING


# ──────────────────────────────────────────────────────
#  DETEKSI POSE
# ──────────────────────────────────────────────────────

def _palm_center(lm_data, w, h):
    """Pusat telapak tangan: rata-rata wrist(0), index_mcp(5), middle_mcp(9), ring_mcp(13)."""
    pts = [(lm_data.landmark[i].x * w, lm_data.landmark[i].y * h)
           for i in (0, 5, 9, 13)]
    return np.mean(pts, axis=0)


def left_covers_nose(lm_data, nose_xy, w, h) -> bool:
    """
    True jika telapak tangan kiri sangat dekat dengan titik hidung.
    Ambang batas = NOSE_COVER_DIST * lebar frame.
    """
    palm = _palm_center(lm_data, w, h)
    dist = np.hypot(palm[0] - nose_xy[0], palm[1] - nose_xy[1])
    return dist < NOSE_COVER_DIST * w


# ──────────────────────────────────────────────────────
#  UI OVERLAY
# ──────────────────────────────────────────────────────

def draw_ui(frame, progress, left_ok, right_ok, state):
    h, w = frame.shape[:2]
    F    = cv2.FONT_HERSHEY_SIMPLEX

    # Judul
    cv2.putText(frame, "CAT POSE DETECTOR", (14, 34), F, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, "CAT POSE DETECTOR", (14, 34), F, 0.7, (0,210,255), 1, cv2.LINE_AA)

    # Badge tangan
    def badge(txt, ok, x, y):
        c = (0,210,80) if ok else (70,70,180)
        s = "OK" if ok else "--"
        cv2.putText(frame, f"{txt}: {s}", (x,y), F, 0.52, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"{txt}: {s}", (x,y), F, 0.52, c,       1, cv2.LINE_AA)

    badge("Kiri  (tutup hidung)",   left_ok,  14, h-52)
    badge("Kanan (wave kiri-kanan)", right_ok, 14, h-30)

    # Progress bar
    bw = int(w*0.55); bh = 10
    bx = (w-bw)//2;  by = h-18
    cv2.rectangle(frame, (bx,by), (bx+bw, by+bh), (35,35,35), -1)
    if progress > 0:
        fill = int(bw * min(progress, 1.0))
        col  = (0,200,255) if progress < 1.0 else (0,255,140)
        cv2.rectangle(frame, (bx,by), (bx+fill, by+bh), col, -1)

    # Pesan
    if state == "playing":
        msg = "Memutar video..."; col = (0,255,140)
    elif state == "cooldown":
        msg = "Ulangi pose untuk main lagi!"; col = (140,140,140)
    elif left_ok and right_ok:
        msg = f"Bagus! Tahan terus... {int(progress*100)}%"; col = (0,220,110)
    elif left_ok:
        msg = "Tangan kiri OK  |  Kanan: wave kiri-kanan"; col = (0,200,255)
    elif right_ok:
        msg = "Tangan kanan OK  |  Kiri: tutup hidung"; col = (0,200,255)
    else:
        msg = "Pose: tangan KIRI tutup hidung + tangan KANAN wave"; col = (170,170,170)

    ts = cv2.getTextSize(msg, F, 0.48, 1)[0]
    tx = max(4, (w-ts[0])//2); ty = by-12
    cv2.putText(frame, msg, (tx+1,ty+1), F, 0.48, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(frame, msg, (tx,ty),     F, 0.48, col,     1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  CAT POSE DETECTOR  v5")
    print("=" * 58)
    print(f"  Video  : {VIDEO_PATH}")
    print(f"  Tahan  : {HOLD_DURATION} detik")
    print()
    print("  POSE:")
    print("   Tangan KIRI  → tutup hidung (dekatkan telapak ke hidung)")
    print("   Tangan KANAN → wave kiri-kanan minimal 2x")
    print()
    print("  Tekan Q untuk keluar.")
    print("=" * 58)

    if not os.path.exists(VIDEO_PATH):
        print(f"\n[!] Video tidak ditemukan: {VIDEO_PATH}")
        print("    Letakkan 'cat_video.mp4' di folder yang sama.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak bisa dibuka.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hands     = _hands_sol.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.65,
    )
    face_mesh = _face_mesh_sol.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    wave         = WaveDetector()
    pose_start   = None
    cooldown_end = 0.0
    state        = "waiting"
    player       = None           # VideoPlayer thread aktif

    # Cache posisi hidung (supaya tidak hilang saat tangan menutupi wajah)
    cached_nose  = None

    mp_draw = mp.solutions.drawing_utils

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now   = time.time()

            # ── Cek apakah video selesai ────────────────────
            if player is not None and player.is_done():
                player   = None
                cooldown_end = now + 1.5
                state    = "cooldown"

            # ── Face Mesh — ambil posisi hidung ─────────────
            mesh_result = face_mesh.process(rgb)
            nose_xy     = cached_nose  # pakai cache dulu

            if mesh_result.multi_face_landmarks:
                lms = mesh_result.multi_face_landmarks[0].landmark
                # Landmark 4 = ujung hidung
                nose_xy     = (lms[4].x * w, lms[4].y * h)
                cached_nose = nose_xy

                # Titik kecil di hidung sebagai panduan (putih)
                nx, ny = int(nose_xy[0]), int(nose_xy[1])
                cv2.circle(frame, (nx, ny), 6,  (255,255,255), -1)
                cv2.circle(frame, (nx, ny), 18, (255,255,255),  1)

            # ── Hand Tracking ───────────────────────────────
            hand_result = hands.process(rgb)
            left_ok     = False
            right_ok    = False
            right_seen  = False

            if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
                for lm_data, handedness in zip(
                    hand_result.multi_hand_landmarks,
                    hand_result.multi_handedness
                ):
                    label = handedness.classification[0].label  # "Left" / "Right"

                    # Gambar landmark PUTIH semua
                    mp_draw.draw_landmarks(
                        frame, lm_data, _hands_sol.HAND_CONNECTIONS,
                        _HAND_LM_STYLE, _HAND_CONN_STYLE,
                    )

                    if label == "Left":
                        if nose_xy is not None:
                            left_ok = left_covers_nose(lm_data, nose_xy, w, h)

                    else:   # Right
                        right_seen = True
                        wx = lm_data.landmark[0].x
                        wave.update(wx)
                        right_ok = wave.is_waving()

            if not right_seen:
                wave.reset()

            # ── State machine ───────────────────────────────
            progress = 0.0

            if player is not None and not player.is_done():
                # Video sedang diputar
                state    = "playing"
                pose_start = None

            elif now < cooldown_end:
                state      = "cooldown"
                pose_start = None

            elif left_ok and right_ok:
                if pose_start is None:
                    pose_start = now
                elapsed  = now - pose_start
                progress = elapsed / HOLD_DURATION
                state    = "holding"

                if progress >= 1.0:
                    # Mulai putar video di thread terpisah
                    if player is None or player.is_done():
                        player = VideoPlayer(VIDEO_PATH)
                        player.start()
                    pose_start = None
                    wave.reset()
                    state = "playing"

            else:
                pose_start = None
                if state not in ("playing", "cooldown"):
                    state = "waiting"

            draw_ui(frame, progress, left_ok, right_ok, state)
            cv2.imshow("Cat Pose Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("\n[INFO] Keluar.")
                if player:
                    player.stop()
                break

    finally:
        cap.release()
        hands.close()
        face_mesh.close()
        cv2.destroyAllWindows()
        print("[INFO] Selesai.")


if __name__ == "__main__":
    main()
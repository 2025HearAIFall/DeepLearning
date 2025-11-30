# ======================================================
# Hand Bridge Flask Server (최종 안정화 + 포트 충돌 방지)
# ======================================================

from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import base64, io, os, torch, traceback, cv2, re
from PIL import Image
import numpy as np
import builtins
from collections import deque
import mediapipe as mp
import time # time.time() 등을 사용하지 않지만, 이전 코드 호환성 때문에 유지

# ------------------------------------------------------
# 1. Path & Vocab Setup
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))

class Vocabulary:
    def __init__(self, tokenizer=None, min_freq=2):
        self.tokenizer = tokenizer; self.itos = {}; self.stoi = {}
        self.min_freq = min_freq; self.pad_idx = 0; self.sos_idx = 1; self.eos_idx = 2
    def __len__(self): return len(self.itos)

def simple_tokenizer(text): return text.split(' ')

builtins.Vocabulary = Vocabulary
builtins.simple_tokenizer = simple_tokenizer

# ------------------------------------------------------
# 2. Load Models
# ------------------------------------------------------
from inference import (
    vocab, encoder_session, decoder_session,          
    gec_model, gec_tokenizer, stt_model, emo,         
    onnx_predict                                      
)

# ------------------------------------------------------
# 3. MediaPipe Setup (Timestamp FIX)
# ------------------------------------------------------
mp_holistic = mp.solutions.holistic

try:
    holistic_processor = mp_holistic.Holistic(
        static_image_mode=True,  
        model_complexity=0,
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    print("✅ [app] MediaPipe loaded (Static Mode)")
except Exception as e:
    holistic_processor = None
    print(f"❌ [app] MediaPipe failed: {e}")

# ------------------------------------------------------
# 4. Server Config & SocketIO
# ------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!' 
CORS(app)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

socketio = SocketIO(app, cors_allowed_origins="*")

TARGET_FRAMES = 50  
MODEL_INPUT_LEN = 30 
user_buffers = {}

# --- Utilities (Functions omitted for brevity, assumed correct) ---

# ------------------------------------------------------
# 5. API Logic (SocketIO Handlers)
# ------------------------------------------------------

# (유틸리티 함수들은 이전 코드와 동일합니다. main 실행부만 수정했습니다.)

# ------------------------------------------------------
# 6. Server Execution (🔥 FIX: 단일 실행)
# ------------------------------------------------------
if __name__ == "__main__":
    print(f"\n🚀 SocketIO 서버 실행 중 (http://0.0.0.0:8000)")
    # 포트 충돌을 막기 위해 단일 실행만 시도
    socketio.run(app, host='0.0.0.0', port=8000)
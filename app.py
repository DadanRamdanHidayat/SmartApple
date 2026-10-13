# =========================
# SMARTAPPLE - VERSI VERCEL
# =========================
import os
import io
import base64
import threading
from datetime import datetime

import numpy as np
import requests
from PIL import Image
from flask import Flask, render_template, request, jsonify, redirect, url_for
from ai_edge_litert.interpreter import Interpreter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# INISIALISASI FLASK
# =========================
app = Flask(__name__)
# Batas body request fungsi Vercel adalah 4.5 MB
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024

# =========================
# LOAD MODEL (TFLite)
# =========================
IMG_SIZE = (224, 224)
interpreter = Interpreter(model_path=os.path.join(BASE_DIR, 'model', 'model.tflite'))
interpreter.allocate_tensors()
_in  = interpreter.get_input_details()[0]['index']
_out = interpreter.get_output_details()[0]['index']
_lock = threading.Lock()   # interpreter tidak thread-safe

def run_model(arr):
    with _lock:
        interpreter.set_tensor(_in, arr)
        interpreter.invoke()
        return interpreter.get_tensor(_out)[0]

# =========================
# LABEL & INFO KELAS
# =========================
CLASS_LABELS = ['Blotch_Apple', 'Normal_Apple', 'Rot_Apple', 'Scab_Apple']

# Optional (biar user friendly)
LABEL_TRANSLATE = {
    'Blotch_Apple': 'Bercak (Blotch)',
    'Normal_Apple': 'Apel Sehat',
    'Rot_Apple': 'Busuk (Rot)',
    'Scab_Apple': 'Kerompeng (Scab)'
}

# =========================
# INFO PER KELAS
# =========================
CLASS_INFO = {
    'Normal_Apple': {
        'kondisi':       'Sehat',
        'estimasi_panen':'Siap dikonsumsi',
        'solusi': {
            'mekanis': ['Pemanenan standar menggunakan sarung tangan kain untuk mencegah luka kuku.'],
            'kimiawi': ['Pemberian lapisan lilin (waxing) food-grade untuk menekan laju respirasi.'],
            'kultur_teknis': ['Pemeliharaan sanitasi kebun dan pemangkasan ranting mati secara berkala.'],
            'pascapanen': ['Penyimpanan dalam cold storage suhu 0-2°C dengan kelembapan relatif (RH) 90-95%.']
        }
    },
    'Blotch_Apple': {
        'kondisi':       'Terdapat bercak',
        'estimasi_panen':'Perlu seleksi',
        'solusi': {
            'mekanis': [
                'Sortasi total di pohon, petik buah bergejala, dan pisahkan radius 10 meter dari area packing.',
                'Pangkas kanopi pohon yang terlalu rapat untuk meningkatkan penetrasi sinar matahari.'
            ],
            'kimiawi': [
                'Semprot fungisida kontak berbahan aktif Mancozeb atau Captan dosis 2g/L air.',
                'Aplikasi rotasi dengan fungisida sistemik golongan triazol (Difenokonazol) interval 7-10 hari.'
            ],
            'kultur_teknis': [
                'Pembersihan gulma bawah tajuk untuk menurunkan kelembapan mikro di sekitar pohon.',
                'Hindari sistem pengairan overhead (sprinkler atas) untuk mencegah penyebaran spora Alternaria lewat percikan air.'
            ],
            'pascapanen': [
                'Alokasikan buah bergejala ringan hanya untuk kebutuhan industri olahan (jus/selai).',
                'Karantina buah di ruang pendingin terpisah, percepat distribusi, dan dilarang disimpan jangka panjang.'
            ]
        }
    },
    'Rot_Apple': {
        'kondisi':       'Busuk',
        'estimasi_panen':'Tidak layak',
        'solusi': {
            'mekanis': [
                'Eradikasi total: petik semua buah busuk di pohon dan kumpulkan buah rontok di tanah.',
                'Pembersihan mumi buah (buah mengering yang menempel di ranting) yang menjadi reservoir utama jamur.'
            ],
            'kimiawi': [
                'Semprot fungisida sistemik berbahan aktif Thiophanate-methyl atau Pyraclostrobin dosis 1.5g/L.',
                'Sterilisasi alat potong/pruning menggunakan alkohol 70% atau larutan pemutih 10% setiap berpindah pohon.'
            ],
            'kultur_teknis': [
                'Pemusnahan material terinfeksi dengan metode pembakaran atau penguburan sedalam minimal 50 cm.',
                'Pemangkasan ekstrem pada ranting/buntut buah yang menunjukkan gejala kanker pohon (canker).'
            ],
            'pascapanen': [
                'Status: TOTAL REJECT. Dilarang keras memasukkan buah ke lini sortasi gudang.',
                'Sterilisasi wadah/keranjang panen yang sempat kontak dengan buah busuk menggunakan larutan klorin.'
            ]
        }
    },
    'Scab_Apple': {
        'kondisi':       'Terinfeksi scab',
        'estimasi_panen':'Perlu penanganan',
        'solusi': {
            'mekanis': [
                'Pangkas tunas/ranting muda yang menunjukkan gejala bercak beludru hijau zaitun.',
                'Gunakan alat mekanis penggaruk daun untuk mengumpulkan seluruh serasah daun gugur di bawah kanopi.'
            ],
            'kimiawi': [
                'Aplikasi fungisida protektan Tembaga Hidroksida (Copper Hydroxide) pada fase green tip (tunas pecah).',
                'Gunakan fungisida sistemik golongan SBI (Steriols Biosynthesis Inhibitors) seperti Flusilazole jika infeksi sudah masuk fase kuratif.'
            ],
            'kultur_teknis': [
                'Aplikasi larutan Urea 5% pada serasah daun di tanah saat musim gugur/sebelum dorman untuk mempercepat dekomposisi daun dan mematikan peritesium jamur.',
                'Perlebar jarak tanam pada replantasi baru untuk mengoptimalkan sirkulasi udara mikro.'
            ],
            'pascapanen': [
                'Sortir buah cacat bentuk (distorsi gabus) ke kelas kualitas rendah (Grade C).',
                'Turunkan kelembapan ruang simpan hingga batas bawah toleransi varietas untuk menghambat perkembangan spora laten selama penyimpanan.'
            ]
        }
    }
}

# ==========================================================
# API KEY GROQ (dari Environment Variable Vercel)
# ==========================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ==========================================================
# FUNGSI REKOMENDASI AI
# ==========================================================
def fallback_text(label_display, kondisi):
    return (f"Hasil deteksi: {label_display} ({kondisi}). Rekomendasi AI sedang tidak "
            f"tersedia, silakan lihat panduan penanganan di bawah.")

def get_ai_recommendation(label_raw, label_display, confidence, kondisi):
    if not GROQ_API_KEY:
        return fallback_text(label_display, kondisi)

    prompt = f"""
Kamu adalah pakar pertanian apel dan penyakit tanaman.

Analisis hasil deteksi AI berikut:

Nama Penyakit : {label_display}
Kondisi Buah  : {kondisi}
Confidence AI : {confidence:.1f}%

Tugas kamu:
1. Jelaskan kondisi apel secara natural.
2. Jelaskan tingkat keparahan penyakit.
3. Berikan rekomendasi penanganan yang praktis.
4. Jelaskan apakah apel masih layak dikonsumsi.
5. Gunakan bahasa Indonesia yang ramah dan profesional.
6. Maksimal 5 kalimat.
7. Langsung berikan saran obat atau kebutuhan lain yang umum digunakan petani beserta contoh mereknya.
8. Sertakan saran pencegahan agar penyakit tidak menyebar.
9. Jangan gunakan format poin.
10. Jangan gunakan pembuka seperti:
   - "Berikut hasil analisis"
   - "Tentu"
   - "Baik"
11. Jawaban harus terasa seperti konsultasi ahli pertanian sungguhan.
"""
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system",
                     "content": "Kamu adalah pakar penyakit tanaman apel dan memberikan jawaban profesional, singkat, dan mudah dipahami petani."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.5
            },
            timeout=8
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"[Groq API error] {type(e).__name__}: {e}")
        return fallback_text(label_display, kondisi)

# =========================
# HISTORY (di memori, TIDAK permanen di Vercel)
# =========================
history_store = []
history_id_counter = 1

# =========================
# ROUTES
# =========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deteksi')
def deteksi():
    return render_template('detectionpage.html')

@app.route('/tentang')
def tentang():
    return render_template('aboutpage.html')

@app.route('/riwayat')
def riwayat():
    stats = {
        'total':  len(history_store),
        'normal': sum(1 for h in history_store if h['label'] == 'Normal_Apple'),
        'blotch': sum(1 for h in history_store if h['label'] == 'Blotch_Apple'),
        'rot':    sum(1 for h in history_store if h['label'] == 'Rot_Apple'),
        'scab':   sum(1 for h in history_store if h['label'] == 'Scab_Apple'),
    }
    return render_template('historypage.html', histories=history_store[::-1], stats=stats)

@app.route('/riwayat/hapus/<int:item_id>')
def hapus_riwayat(item_id):
    global history_store
    history_store = [h for h in history_store if h['id'] != item_id]
    return redirect(url_for('riwayat'))

@app.route('/riwayat/hapus-semua')
def hapus_semua():
    global history_store
    history_store = []
    return redirect(url_for('riwayat'))

# =========================
# VALIDASI FILE
# =========================
ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# =========================
# PREPROCESS (tanpa menyimpan file)
# =========================
def load_image(file_storage):
    img = Image.open(file_storage.stream).convert('RGB')
    # thumbnail JPEG kecil (data URI) untuk halaman riwayat
    th = img.copy()
    th.thumbnail((320, 320))
    buf = io.BytesIO()
    th.save(buf, format='JPEG', quality=70)
    thumb = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
    # input model: 224x224, skala 0-1
    arr = np.asarray(img.resize(IMG_SIZE), dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0), thumb

# =========================
# PREDICT
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    global history_id_counter

    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Format file tidak valid'}), 400

    try:
        arr, thumb = load_image(file)
        prediction  = run_model(arr)
        class_index = int(np.argmax(prediction))

        label_raw  = CLASS_LABELS[class_index]
        label      = LABEL_TRANSLATE.get(label_raw, label_raw)
        confidence = float(prediction[class_index])
        info       = CLASS_INFO.get(label_raw, {})

        ai_recommendation = get_ai_recommendation(
            label_raw, label, confidence * 100, info.get('kondisi')
        )

        filename = datetime.now().strftime('%Y%m%d_%H%M%S_') + os.path.basename(file.filename)
        history_store.append({
            'id':         history_id_counter,
            'filename':   filename,
            'thumb':      thumb,
            'label':      label_raw,
            'confidence': confidence,
            'created_at': datetime.now()
        })
        history_id_counter += 1

        return jsonify({
            'label':             label,
            'confidence':        confidence,
            'estimasi_panen':    info.get('estimasi_panen'),
            'kondisi':           info.get('kondisi'),
            'solusi':            info.get('solusi'),
            'ai_recommendation': ai_recommendation,
            'filename':          filename
        })

    except Exception as e:
        print(f"[Predict error] {type(e).__name__}: {e}")
        return jsonify({'error': 'Gagal memproses gambar. Pastikan file adalah foto yang valid.'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Ukuran file maksimal 4 MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('notfound.html'), 404

if __name__ == '__main__':
    app.run(debug=False)

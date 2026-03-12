from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import random
import traceback
from datetime import datetime
import shutil
import subprocess

# 🌟 นำเข้าเครื่องมือสำหรับจัดการภาพและเสียง
import numpy as np
import tensorflow as tf
import joblib
import librosa
import noisereduce as nr
import soundfile as sf
from werkzeug.utils import secure_filename
import re
import cv2
from deepface import DeepFace
from collections import Counter

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False
    print("⚠️ Warning: yt-dlp is not installed.")

app = Flask(__name__)
# อนุญาตให้ Frontend จาก Vercel หรือที่อื่นๆ ยิงข้อมูลเข้ามาได้
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================
# 1. DATABASE CONFIG
# ==========================================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///vibesense.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

ADMIN_PASSWORD = "admin1234"

class Product(db.Model):
    id = db.Column(db.String(10), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    brand = db.Column(db.String(50))
    price = db.Column(db.String(20))
    rating = db.Column(db.Float, default=4.9)
    image = db.Column(db.String(200))
    tag = db.Column(db.String(50))
    mood = db.Column(db.String(20)) 
    detail = db.Column(db.Text)
    vibe_logic = db.Column(db.Text)

class SystemStats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    total_scans = db.Column(db.Integer, default=0)
    happy_count = db.Column(db.Integer, default=0)
    sad_count = db.Column(db.Integer, default=0)
    angry_count = db.Column(db.Integer, default=0)
    neutral_count = db.Column(db.Integer, default=0)

# ==========================================
# 2. LOAD AI ASSETS
# ==========================================
MODEL_PATH = 'models/emotion_model.h5'
CLASSES_PATH = 'models/classes.npy'
SCALER_PATH = 'models/scaler.pkl'

emotion_model = None
class_labels = None
scaler = None

try:
    print("🧠 Loading AI Assets...")
    if os.path.exists(MODEL_PATH):
        emotion_model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ AI Model Loaded! Expected Input Shape: {emotion_model.input_shape}")
    if os.path.exists(CLASSES_PATH):
        class_labels = np.load(CLASSES_PATH, allow_pickle=True)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("✅ Scaler Loaded!")
except Exception as e:
    print(f"⚠️ AI Loading Error: {e}")

# ==========================================
# 3. API ROUTES
# ==========================================
@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    data = request.json
    if data.get('password') == ADMIN_PASSWORD:
        return jsonify({'status': 'success', 'token': 'secure-admin-session'})
    return jsonify({'status': 'error', 'message': 'Invalid Password'}), 401

@app.route('/api/admin/stats', methods=['GET'])
def get_stats():
    stats = SystemStats.query.first()
    if not stats:
        stats = SystemStats(id=1)
        db.session.add(stats)
        db.session.commit()
    return jsonify({
        'totalScans': stats.total_scans,
        'distribution': {
            'happy': stats.happy_count, 'sad': stats.sad_count,
            'angry': stats.angry_count, 'neutral': stats.neutral_count
        }
    })

@app.route('/api/products', methods=['GET', 'POST'])
def handle_products():
    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            mood = request.form.get('mood', 'neutral')
            
            new_id = f"{mood[0]}{random.randint(100,999)}"
            ext = os.path.splitext(file.filename)[1]
            if not ext: ext = '.jpg'
            filename = f"{new_id}{ext}"
            
            save_path = os.path.join('static', 'products', filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            file.save(save_path)
            
            new_p = Product(
                id=new_id,
                name=request.form.get('name'),
                brand=request.form.get('brand', 'VibeBrand'),
                price=request.form.get('price'),
                mood=mood,
                image=filename,
                tag=request.form.get('tag', 'New Arrival'),
                detail=request.form.get('detail', 'รายละเอียดสินค้าใหม่'),
                vibe_logic=request.form.get('vibe_logic', 'AI คัดสรรมาเพื่อคุณ')
            )
        else:
            data = request.json
            new_id = f"{data['mood'][0]}{random.randint(100,999)}"
            new_p = Product(
                id=new_id,
                name=data['name'],
                brand=data.get('brand', 'VibeBrand'),
                price=data['price'],
                mood=data['mood'],
                image=data.get('image', 'n1.jpg'),
                tag=data.get('tag', 'New Arrival'),
                detail=data.get('detail', 'รายละเอียดสินค้าใหม่'),
                vibe_logic=data.get('vibe_logic', 'AI คัดสรรมาเพื่อคุณ')
            )
        
        db.session.add(new_p)
        db.session.commit()
        return jsonify({'status': 'success', 'id': new_id})

    products = Product.query.all()
    # 🌟 ทำให้ URL เป็น Dynamic อัตโนมัติตาม IP หรือ Domain ที่เครื่องเปิดอยู่
    return jsonify([{
        'id': p.id, 'name': p.name, 'brand': p.brand, 'price': p.price,
        'rating': p.rating, 
        'image': f"{request.host_url.rstrip('/')}/static/products/{p.image}",
        'tag': p.tag, 'mood': p.mood, 'detail': p.detail, 'vibeReason': p.vibe_logic
    } for p in products])

@app.route('/api/products/<id>', methods=['DELETE'])
def delete_product(id):
    p = Product.query.get(id)
    if p:
        db.session.delete(p)
        db.session.commit()
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'}), 404

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    if not os.path.exists('temp'): os.makedirs('temp')
    filepath = None
    temp_demucs_folder = os.path.join('temp', f"demucs_{random.randint(1000,9999)}")
    temp_clean_wav = os.path.join('temp', f"clean_{random.randint(1000,9999)}.wav")
    
    try:
        video_url = request.form.get('url')
        
        if video_url:
            if not YTDLP_AVAILABLE:
                return jsonify({'error': 'yt-dlp is not installed on server'}), 500
            
            video_url = re.sub(r'youtube\.com/shorts/([a-zA-Z0-9_-]+)', r'youtube.com/watch?v=\1', video_url)
            print(f"🔗 Downloading video from URL: {video_url}")
            ydl_opts = {
                'outtmpl': os.path.join('temp', 'downloaded_%(id)s.%(ext)s'),
                'format': 'worst[ext=mp4]/worst', 
                'noplaylist': True,
                'quiet': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                filepath = ydl.prepare_filename(info)
                
        elif 'file' in request.files:
            file = request.files['file']
            original_ext = os.path.splitext(file.filename)[1]
            if not original_ext:
                original_ext = '.mp4' 
            safe_filename = f"upload_{int(datetime.now().timestamp())}_{random.randint(1000,9999)}{original_ext}"
            filepath = os.path.join('temp', safe_filename)
            file.save(filepath)
        else:
            return jsonify({'error': 'No video file or URL provided'}), 400

        # --- AI PART 1: AUDIO ANALYSIS ---
        final_emotion = 'neutral'
        final_confidence = 0.0
        audio_emotion = None
        audio_conf = 0.0
        face_emotion = None
        face_conf = 0.0

        if emotion_model and class_labels is not None and scaler is not None:
            try:
                print("🎧 [1/3] Extracting Vocals using Demucs (Please wait)...")
                command = f'demucs -n htdemucs --two-stems=vocals -o "{temp_demucs_folder}" "{filepath}"'
                subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                video_name_no_ext = os.path.splitext(os.path.basename(filepath))[0]
                vocal_path = os.path.join(temp_demucs_folder, "htdemucs", video_name_no_ext, "vocals.wav")
                
                if os.path.exists(vocal_path):
                    print("🧹 [2/3] Cleaning audio with NoiseReduce...")
                    data, rate = librosa.load(vocal_path, sr=None)
                    clean_audio = nr.reduce_noise(y=data, sr=rate, stationary=True)
                    sf.write(temp_clean_wav, clean_audio, rate)
                    y, sr = librosa.load(temp_clean_wav, sr=None)
                else:
                    raise Exception("Demucs Failed")
            except Exception as audio_err:
                print(f"⚠️ Demucs Failed: {audio_err}. Switching to Direct Librosa...")
                y, sr = librosa.load(filepath, sr=None)

            try:
                if len(y) > 0:
                    segment_duration = 3.0
                    samples_per_segment = int(segment_duration * sr)
                    step_size = int(1.5 * sr)
                    audio_predictions = []
                    
                    for i in range(0, len(y) - samples_per_segment + 1, step_size):
                        y_segment = y[i : i + samples_per_segment]
                        mfcc = np.mean(librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=83).T, axis=0)
                        features = mfcc.reshape(1, 83)
                        scaled_features = scaler.transform(features)
                        preds = emotion_model.predict(scaled_features, verbose=0)
                        score_index = np.argmax(preds[0])
                        raw_emotion = str(class_labels[score_index]).lower().strip()
                        pred_emotion = 'neutral'
                        if 'hap' in raw_emotion: pred_emotion = 'happy'
                        elif 'sad' in raw_emotion: pred_emotion = 'sad'
                        elif 'ang' in raw_emotion: pred_emotion = 'angry'
                        audio_predictions.append(pred_emotion)
                    
                    if not audio_predictions:
                        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=83).T, axis=0)
                        features = mfcc.reshape(1, 83)
                        scaled_features = scaler.transform(features)
                        preds = emotion_model.predict(scaled_features, verbose=0)
                        score_index = np.argmax(preds[0])
                        raw_emotion = str(class_labels[score_index]).lower().strip()
                        if 'hap' in raw_emotion: audio_emotion = 'happy'
                        elif 'sad' in raw_emotion: audio_emotion = 'sad'
                        elif 'ang' in raw_emotion: audio_emotion = 'angry'
                        else: audio_emotion = 'neutral'
                        audio_conf = float(preds[0][score_index]) * 100
                    else:
                        counts = Counter(audio_predictions)
                        most_common = counts.most_common(1)[0]
                        audio_emotion = most_common[0]
                        audio_conf = (most_common[1] / len(audio_predictions)) * 100
            except Exception as ve:
                print(f"⚠️ Audio Analysis Error: {ve}")

        # --- AI PART 2: VISUAL ANALYSIS ---
        try:
            print("📸 Starting DeepFace Visual Analysis...")
            cap = cv2.VideoCapture(filepath)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_to_check = [total_frames//4, total_frames//2, (total_frames*3)//4] if total_frames > 15 else [total_frames//2]
                
            face_emotions = []
            for frame_idx in frames_to_check:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = cap.read()
                if success:
                    try:
                        objs = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv', silent=True)
                        result = objs[0] if isinstance(objs, list) else objs
                        deepface_emotion = result['dominant_emotion'].lower()
                        df_mapped = 'neutral'
                        if deepface_emotion in ['happy']: df_mapped = 'happy'
                        elif deepface_emotion in ['sad']: df_mapped = 'sad'
                        elif deepface_emotion in ['angry', 'disgust', 'fear']: df_mapped = 'angry'
                        elif deepface_emotion in ['surprise']: df_mapped = 'happy' 
                        face_emotions.append(df_mapped)
                    except: pass
            cap.release()

            if face_emotions:
                counts = Counter(face_emotions)
                most_common = counts.most_common(1)[0]
                face_emotion = most_common[0]
                face_conf = (most_common[1] / len(face_emotions)) * 100
        except Exception as visual_err:
            print(f"⚠️ Visual Analysis Failed: {visual_err}")

        # --- AI PART 3: SENSOR FUSION ---
        if audio_emotion and face_emotion:
            if audio_conf >= face_conf: final_emotion, final_confidence = audio_emotion, audio_conf
            else: final_emotion, final_confidence = face_emotion, face_conf
        elif audio_emotion: final_emotion, final_confidence = audio_emotion, audio_conf
        elif face_emotion: final_emotion, final_confidence = face_emotion, face_conf
        else: raise Exception("AI ไม่สามารถสกัดข้อมูลจากวิดีโอนี้ได้")

        # 📊 Update System Stats
        stats = SystemStats.query.first()
        if not stats: stats = SystemStats(id=1)
        stats.total_scans += 1
        if final_emotion == 'happy': stats.happy_count += 1
        elif final_emotion == 'sad': stats.sad_count += 1
        elif final_emotion == 'angry': stats.angry_count += 1
        else: stats.neutral_count += 1
        db.session.add(stats)
        db.session.commit()

        if filepath and os.path.exists(filepath): os.remove(filepath)
        if temp_clean_wav and os.path.exists(temp_clean_wav): os.remove(temp_clean_wav)
        if os.path.exists(temp_demucs_folder): shutil.rmtree(temp_demucs_folder, ignore_errors=True)
        
        return jsonify({
            'status': 'success', 'emotion': final_emotion, 
            'confidence': f"{final_confidence:.2f}%",
            'timestamp': datetime.now().strftime('%d.%m.%Y // %H:%M')
        })

    except Exception as e:
        if filepath and os.path.exists(filepath): os.remove(filepath)
        if temp_clean_wav and os.path.exists(temp_clean_wav): os.remove(temp_clean_wav)
        if os.path.exists(temp_demucs_folder): shutil.rmtree(temp_demucs_folder, ignore_errors=True)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/static/products/<filename>')
def serve_image(filename):
    return send_from_directory('static/products', filename)

def seed_products():
    if Product.query.first() is None:
        items = [
            Product(id='h1', name='Party Poppers', brand='PartyShop', price='฿150', rating=4.8, image='h1.jpg', tag='Party', mood='happy', detail='พลุกระดาษสีสันสดใส ขนาดกะทัดรัด ใช้งานง่าย', vibe_logic='สีสันและการเฉลิมฉลองจะช่วยเพิ่มระดับโดปามีน เสริมสร้างอารมณ์เชิงบวก'),
            Product(id='h2', name='Polaroid Go', brand='Polaroid', price='฿4,290', rating=4.7, image='h2.jpg', tag='Instant', mood='happy', detail='กล้องอินสแตนท์ขนาดเล็ก พกพาสะดวก', vibe_logic='การบันทึกภาพความสุขช่วยสร้างความทรงจำระยะยาวที่ส่งผลดีต่อจิตใจ'),
            Product(id='h3', name='Colorful Socks Set', brand='HappySocks', price='฿350', rating=4.9, image='h3.jpg', tag='Fashion', mood='happy', detail='เซ็ตถุงเท้าลายกราฟิกสีสันสดใส', vibe_logic='การสวมใส่เสื้อผ้าที่มีสีสันสดใสสามารถกระตุ้นอารมณ์ร่าเริงได้'),
            Product(id='h4', name='UNO Card Game', brand='Mattel', price='฿250', rating=4.9, image='h4.jpg', tag='Game', mood='happy', detail='การ์ดเกมคลาสสิกสำหรับเล่นกับครอบครัว', vibe_logic='กิจกรรมกลุ่มที่สร้างปฏิสัมพันธ์ทางสังคม ช่วยหลั่งฮอร์โมนแห่งความผูกพัน'),
            Product(id='h5', name='Mini Bluetooth Speaker', brand='JBL', price='฿1,290', rating=4.8, image='h5.jpg', tag='Audio', mood='happy', detail='ลำโพงบลูทูธพกพากันน้ำ ให้เสียงเบสหนักแน่น', vibe_logic='ดนตรีจังหวะสนุกสนานเป็นตัวเร่งปฏิกิริยาอารมณ์ดีได้อย่างรวดเร็ว'),
            Product(id='h6', name='Ice Cream Maker', brand='HomeCook', price='฿890', rating=4.5, image='h6.jpg', tag='Appliance', mood='happy', detail='เครื่องทำไอศกรีมโฮมเมด ทำง่ายใน 20 นาที', vibe_logic='การทำกิจกรรมที่ได้ผลลัพธ์เป็นของหวานเชื่อมโยงกับความรู้สึกให้รางวัลตัวเอง'),
            Product(id='h7', name='Gummy Bears Jar', brand='Haribo', price='฿180', rating=4.7, image='h7.jpg', tag='Snack', mood='happy', detail='เยลลี่หมีรสผลไม้รวม', vibe_logic='ความหวานในปริมาณที่พอเหมาะกระตุ้นความรู้สึกสดชื่นได้อย่างรวดเร็ว'),
            Product(id='h8', name='Karaoke Mic', brand='K-Pop', price='฿450', rating=4.6, image='h8.jpg', tag='Entertain', mood='happy', detail='ไมโครโฟนคาราโอเกะบลูทูธไร้สาย', vibe_logic='การร้องเพลงเป็นการปลดปล่อยอารมณ์และเสริมความเสถียรของความสุข'),
            Product(id='s1', name='Extra Soft Tissues', brand='Kleenex', price='฿180', rating=4.9, image='s1.jpg', tag='Care', mood='sad', detail='กระดาษทิชชู่หนา 3 ชั้น ผสมโลชั่น', vibe_logic='สัมผัสที่นุ่มนวลช่วยปลอบประโลมจิตใจในยามอ่อนไหว'),
            Product(id='s2', name='Cozy Fleece Blanket', brand='MUJI', price='฿590', rating=4.8, image='s2.jpg', tag='Comfort', mood='sad', detail='ผ้าห่มฟลีซเนื้อนุ่ม น้ำหนักเบา', vibe_logic='ความอบอุ่นคล้ายการถูกกอด ช่วยสร้างความรู้สึกปลอดภัยและลดความกังวล'),
            Product(id='s3', name='Premium Hot Cocoa', brand='Van Houten', price='฿350', rating=4.9, image='s3.jpg', tag='Beverage', mood='sad', detail='ผงโกโก้แท้ รสชาติเข้มข้น', vibe_logic='เครื่องดื่มอุ่นๆ ช่วยกระตุ้นการสร้างเซโรโทนิน ลดความเศร้าได้'),
            Product(id='s4', name='Lavender Candle', brand='Yankee', price='฿450', rating=4.7, image='s4.jpg', tag='Aroma', mood='sad', detail='เทียนหอมกลิ่นลาเวนเดอร์', vibe_logic='กลิ่นลาเวนเดอร์ช่วยลดอัตราการเต้นของหัวใจและคลายความเศร้าหมอง'),
            Product(id='s5', name='Fluffy Teddy Bear', brand='Miniso', price='฿650', rating=4.9, image='s5.jpg', tag='Hug', mood='sad', detail='ตุ๊กตาหมีขนนุ่มฟู', vibe_logic='การกอดสิ่งของนุ่มๆ ช่วยหลั่งฮอร์โมนแห่งความรัก ต้านทานความเศร้า'),
            Product(id='s6', name='Sleepy Bath Bomb', brand='Lush', price='฿320', rating=4.8, image='s6.jpg', tag='Relax', mood='sad', detail='สบู่ทำฟองแช่น้ำ กลิ่นหอมผ่อนคลาย', vibe_logic='การแช่น้ำอุ่นส่งเสริมการนอนหลับลึกเพื่อฟื้นฟูจิตใจ'),
            Product(id='s7', name='Lo-Fi Vinyl Record', brand='Indie', price='฿850', rating=4.8, image='s7.jpg', tag='Music', mood='sad', detail='แผ่นเสียงรวมเพลง Lo-Fi', vibe_logic='จังหวะดนตรีสม่ำเสมอช่วยปรับคลื่นสมอง คืนความสงบให้จิตใจ'),
            Product(id='s8', name='Warm Eye Mask', brand='MegRhythm', price='฿220', rating=4.7, image='s8.jpg', tag='Relief', mood='sad', detail='แผ่นประคบตาอุ่น', vibe_logic='ความร้อนบริเวณดวงตาช่วยลดความตึงเครียดของกล้ามเนื้อ'),
            Product(id='a1', name='Stress Relief Ball', brand='Smiggle', price='฿99', rating=4.6, image='a1.jpg', tag='Relief', mood='angry', detail='ลูกบอลเจลบีบ คืนตัวเร็ว', vibe_logic='การบีบระบายพลังงานส่วนเกิน ช่วยลดระดับความก้าวร้าวได้อย่างเป็นรูปธรรม'),
            Product(id='a2', name='Mini Punching Bag', brand='DeskFit', price='฿350', rating=4.7, image='a2.jpg', tag='Workout', mood='angry', detail='กระสอบทรายจิ๋วตั้งโต๊ะ', vibe_logic='การระบายความโกรธผ่านการออกกำลังกายช่วยปรับสมดุลอะดรีนาลีน'),
            Product(id='a3', name='Foam Earplugs', brand='3M', price='฿85', rating=4.8, image='a3.jpg', tag='Isolation', mood='angry', detail='จุกอุดหูโฟม ลดเสียงรบกวน', vibe_logic='การตัดสิ่งเร้าจากภายนอกช่วยหยุดการกระตุ้นประสาทที่เพิ่มความหงุดหงิด'),
            Product(id='a4', name='Chamomile Tea', brand='Twinings', price='฿280', rating=4.9, image='a4.jpg', tag='Calm', mood='angry', detail='ชาคาโมมายล์บริสุทธิ์', vibe_logic='คาโมมายล์ช่วยส่งเสริมความสงบและลดความฉุนเฉียว'),
            Product(id='a5', name='Fidget Spinner', brand='ToysR', price='฿120', rating=4.5, image='a5.jpg', tag='Focus', mood='angry', detail='ของเล่นหมุนด้วยนิ้วมือ', vibe_logic='การเคลื่อนไหวซ้ำๆ ช่วยเบี่ยงเบนความสนใจจากความโกรธ'),
            Product(id='a6', name='Cooling Gel Patch', brand='KoolFever', price='฿150', rating=4.8, image='a6.jpg', tag='Chill', mood='angry', detail='แผ่นเจลให้ความเย็น', vibe_logic='การลดอุณหภูมิร่างกายช่วยลดการตื่นตัวของระบบประสาทที่ทำงานหนัก'),
            Product(id='a7', name='Herbal Inhaler', brand='Hong Thai', price='฿45', rating=4.9, image='a7.jpg', tag='Scent', mood='angry', detail='ยาดมสมุนไพร กลิ่นหอมเย็น', vibe_logic='กลิ่นเย็นช่วยกระตุ้นการหายใจลึกขึ้น ซึ่งเป็นกลไกระงับอารมณ์โกรธ'),
            Product(id='a8', name='Rugged Phone Case', brand='UAG', price='฿890', rating=4.8, image='a8.jpg', tag='Protect', mood='angry', detail='เคสกันกระแทกมาตรฐานทหาร', vibe_logic='ป้องกันความเสียหายของทรัพย์สินในช่วงที่อารมณ์ขาดการควบคุม'),
            Product(id='n1', name='A4 Grid Notebook', brand='MUJI', price='฿95', rating=4.8, image='n1.jpg', tag='Notes', mood='neutral', detail='สมุดโน้ตตีเส้นตาราง', vibe_logic='ในสภาวะอารมณ์ปกติ สมองจะพร้อมรับข้อมูล สมุดโน้ตช่วยดึงความ Productive ออกมา'),
            Product(id='n2', name='Smooth Gel Pen Set', brand='Pentel', price='฿150', rating=4.9, image='n2.jpg', tag='Writing', mood='neutral', detail='ปากกาเจลเขียนลื่น', vibe_logic='เครื่องมือคุณภาพช่วยให้การทำงานราบรื่น ไม่เกิดความสะดุด'),
            Product(id='n3', name='Insulated Bottle', brand='HydroFlask', price='฿1,250', rating=4.9, image='n3.jpg', tag='Utility', mood='neutral', detail='กระบอกน้ำสแตนเลสเก็บอุณหภูมิ', vibe_logic='การรักษาระดับน้ำในร่างกายเป็นพื้นฐานสำคัญในการรักษาสภาวะอารมณ์'),
            Product(id='n4', name='Desk Organizer', brand='IKEA', price='฿290', rating=4.7, image='n4.jpg', tag='Organize', mood='neutral', detail='กล่องจัดระเบียบโต๊ะทำงาน', vibe_logic='สภาพแวดล้อมที่เป็นระเบียบช่วยลดภาระสมอง ทำให้จดจ่อกับงานได้ดีขึ้น'),
            Product(id='n5', name='Wireless Mouse', brand='Logitech', price='฿550', rating=4.8, image='n5.jpg', tag='Tech', mood='neutral', detail='เมาส์ไร้สายคลิกเงียบ', vibe_logic='เพิ่มประสิทธิภาพในการทำงาน ลดความเมื่อยล้า รักษาสภาวะอารมณ์ปกติ'),
            Product(id='n6', name='Canvas Tote Bag', brand='Uniqlo', price='฿250', rating=4.6, image='n6.jpg', tag='Lifestyle', mood='neutral', detail='กระเป๋าผ้าแคนวาสมินิมอล', vibe_logic='ความเรียบง่ายตอบโจทย์สภาวะจิตใจที่ไม่ได้ต้องการสิ่งเร้าเป็นพิเศษ'),
            Product(id='n7', name='Drip Coffee Bags', brand='Roots', price='฿220', rating=4.8, image='n7.jpg', tag='Energy', mood='neutral', detail='กาแฟดริปแบบซอง', vibe_logic='คาเฟอีนช่วยเพิ่มระดับความตื่นตัวและรักษาความคมชัดของสมาธิ'),
            Product(id='n8', name='Cable Clips', brand='Baseus', price='฿60', rating=4.5, image='n8.jpg', tag='Gadget', mood='neutral', detail='ที่รัดสายไฟซิลิโคน', vibe_logic='การจัดการสิ่งเล็กๆ ช่วยป้องกันไม่ให้อารมณ์ปกติแปรเปลี่ยนเป็นความหงุดหงิด')
        ]
        db.session.bulk_save_objects(items)
        db.session.commit()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        seed_products() 
    # 🌟 อัปเดตสำหรับการ Deploy: รับค่า Port จากระบบ หรือใช้พอร์ต 10000
    port = int(os.environ.get('PORT', 10000))
    print(f"🚀 VibeSense Backend is ready for Production on Port {port}!")
    # เอา ssl_context ออก เพราะระบบ Cloud จะทำ HTTPS ให้เราอัตโนมัติ
    app.run(host='0.0.0.0', debug=False, port=port)
# ใช้ Python 3.10 เป็นฐาน (เบาและเสถียร)
FROM python:3.10-slim

# ตั้งค่าโฟลเดอร์ทำงานในเซิร์ฟเวอร์
WORKDIR /app

# 🌟 ลงโปรแกรมพื้นฐานที่ AI ต้องใช้ (ffmpeg สำหรับแยกเสียงร้อง)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# ก๊อปปี้ไฟล์ requirements และติดตั้ง Library
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ก๊อปปี้โค้ดทั้งหมด (app.py, models, etc.) เข้าไปในเซิร์ฟเวอร์
COPY . .

# สั่งรัน AI ด้วย gunicorn พร้อมตั้งเวลา Timeout 5 นาที
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1 --threads 2
# 🟢 關鍵：強制使用 Python 3.10 (內建 audioop，完美支援 Gradio 4)
FROM python:3.10

# 設定工作目錄
WORKDIR /app

# 1. 安裝系統依賴 (ffmpeg, espeak 等)
COPY packages.txt .
RUN apt-get update && xargs -r -a packages.txt apt-get install -y && rm -rf /var/lib/apt/lists/*

# 2. 安裝 Python 依賴
COPY requirements.txt .
# 升級 pip 以防萬一
RUN pip install --no-cache-dir --upgrade pip
# 安裝您的套件
RUN pip install --no-cache-dir -r requirements.txt

# 3. 複製程式碼
COPY . .

# 4. 設定權限 (Hugging Face 建議)
RUN chmod -R 777 /app

# 5. 啟動指令
CMD ["python", "app.py"]

# Linux Installation & Troubleshooting Guide

## Quick Fix for libGL.so.1 Error

The error `libGL.so.1: cannot open shared object file` occurs because OpenCV requires OpenGL libraries on Linux.

### Solution 1: Install Missing Libraries (Recommended)
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

# CentOS/RHEL/Fedora
sudo yum install -y mesa-libGL glib2

# Alpine Linux
sudo apk add mesa-gl glib
```

### Solution 2: Use Headless OpenCV (Already Updated)
```bash
# Reinstall with headless version
pip uninstall opencv-python
pip install opencv-python-headless
```

### Solution 3: Install Tesseract OCR
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng

# CentOS/RHEL/Fedora
sudo yum install -y tesseract tesseract-langpack-eng

# Verify installation
tesseract --version
```

## Complete Linux Setup

### 1. Install System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils
```

### 2. Create Virtual Environment
```bash
cd /home/adminuser/pdf-summarizer
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
```bash
# Groq API Key
export GROQ_API_KEY="your_api_key_here"

# Email Configuration (optional)
export EMAIL_SENDER="your_email@gmail.com"
export EMAIL_PASSWORD="your_app_password"
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
```

### 5. Run Streamlit App
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## Docker Alternative (Recommended for Production)

### Dockerfile
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
```

### Docker Commands
```bash
# Build image
docker build -t pdf-summarizer .

# Run container
docker run -p 8501:8501 \
    -e GROQ_API_KEY="your_api_key_here" \
    pdf-summarizer
```

## Troubleshooting

### ImportError: libGL.so.1
```bash
# Install OpenGL libraries
sudo apt-get install -y libgl1-mesa-glx libglu1-mesa

# Or use headless OpenCV
pip install opencv-python-headless
```

### Tesseract Not Found
```bash
# Install Tesseract
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng

# Check installation
which tesseract
tesseract --version
```

### Permission Denied
```bash
# Give proper permissions
chmod +x app.py
chmod -R 755 /home/adminuser/pdf-summarizer
```

### Port Already in Use
```bash
# Kill existing process
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run app.py --server.port 8502
```

## Environment Variables for Production

### Create .env file
```bash
cat > .env << EOF
GROQ_API_KEY=your_api_key_here
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EOF
```

### Load .env file
```bash
# In your shell startup script (.bashrc or .zshrc)
source /home/adminuser/pdf-summarizer/.env

# Or load manually
export $(cat .env | xargs)
```

## Testing the Installation

### 1. Test Dependencies
```python
# test_dependencies.py
import sys
print("Python version:", sys.version)

try:
    import pymupdf
    print("✅ PyMuPDF: OK")
except ImportError as e:
    print("❌ PyMuPDF:", e)

try:
    import cv2
    print("✅ OpenCV: OK")
except ImportError as e:
    print("❌ OpenCV:", e)

try:
    import pytesseract
    print("✅ Tesseract: OK")
except ImportError as e:
    print("❌ Tesseract:", e)

try:
    import streamlit
    print("✅ Streamlit: OK")
except ImportError as e:
    print("❌ Streamlit:", e)
```

### 2. Run Test
```bash
python test_dependencies.py
streamlit run app.py
```

The app should now work without the `libGL.so.1` error!

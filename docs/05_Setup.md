# Setup & Installation Guide

## Daftar Isi
1. [Persyaratan Sistem](#persyaratan-sistem)
2. [Instalasi Lokal](#instalasi-lokal)
3. [Konfigurasi Environment](#konfigurasi-environment)
4. [Running Aplikasi](#running-aplikasi)
5. [Deployment ke Cloud](#deployment-ke-cloud)
6. [Troubleshooting](#troubleshooting)

---

## Persyaratan Sistem

### **Hardware Requirements**

#### **Minimum Configuration** (untuk inference)
```
CPU:      Intel i5 / AMD Ryzen 5 (4+ cores)
RAM:      8 GB
Storage:  10 GB (models + data)
GPU:      Optional
Network:  Internet untuk Streamlit Cloud
```

#### **Recommended Configuration** (untuk training)
```
CPU:      Intel i7/i9 or AMD Ryzen 7/9 (8+ cores)
RAM:      32 GB
Storage:  50 GB (models + intermediate data)
GPU:      NVIDIA A100 / V100 / RTX 3080+ (16GB+ VRAM)
CUDA:     11.8+
Network:  High-speed untuk data transfer
```

### **Software Requirements**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9 | 3.10 - 3.11 |
| CUDA | 11.0 | 11.8 |
| cuDNN | 8.2 | 8.6+ |
| conda | 4.10 | Latest |

### **OS Support**

✅ **Tested on**:
- Ubuntu 20.04 LTS
- Ubuntu 22.04 LTS
- WSL2 (Windows Subsystem for Linux)
- macOS 12+ (Intel & Apple Silicon)

⚠️ **Known Issues**:
- GPU support limited on macOS (use CPU)
- Windows requires WSL2 for GPU access

---

## Instalasi Lokal

### **Step 1: Install Conda**

```bash
# Download Miniconda (recommended)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh

# Restart terminal atau run
source ~/.bashrc

# Verify
conda --version
```

### **Step 2: Clone Repository**

```bash
git clone https://github.com/yourusername/dsproject_agriculturepredict.git
cd dsproject_agriculturepredict
```

### **Step 3: Create Conda Environment**

#### **Option A: Dari environment-local.yml (Recommended)**

```bash
# Create environment dengan exact dependencies
conda env create -f environment-local.yml

# Activate
conda activate rapids-24.10

# Verify
python -c "import torch; print(torch.__version__)"
```

#### **Option B: Manual Setup (jika environment-local.yml tidak ada)**

```bash
# Create environment with Python 3.10
conda create -n rapids-24.10 python=3.10 -y

# Activate
conda activate rapids-24.10

# Install RAPIDS
conda install -c nvidia -c conda-forge cudf=24.10 cudatoolkit=11.8 -y

# Install PyTorch dengan CUDA support
conda install pytorch::pytorch pytorch::pytorch-cuda=11.8 -c pytorch -y

# Install dependencies
pip install -r requirements.txt
```

### **Step 4: Verify Installation**

```bash
# Test PyTorch & GPU
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Test cuDF & GPU
python -c "
import cudf
print(f'cuDF version: {cudf.__version__}')
print(f'GPU device: {cudf.current_device()}')"

# Test Streamlit
streamlit --version
```

Expected output:
```
PyTorch version: 2.0.0
CUDA available: True
Device: NVIDIA A100 (or your GPU)
cuDF version: 24.10
GPU device: 0
Streamlit, version 1.28.0
```

---

## Konfigurasi Environment

### **Environment Variables**

Create file `.env` di project root:

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0           # Select GPU device (0, 1, 2, ...)
CUDA_DEVICE_ORDER=PCI_BUS_ID     # Alternative GPU ordering

# PyTorch Configuration
TORCH_HOME=/path/to/models       # Cache directory
MKL_THREADING_LAYER=GNU          # Faster on multi-core CPU

# Streamlit Configuration
STREAMLIT_LOGGER_LEVEL=info
STREAMLIT_CLIENT_MAXMESSAGESIZE=200  # For large dataframes

# Data Configuration
DATA_RAW_PATH=./data/raw
DATA_PROCESSED_PATH=./data/processed
MODEL_PATH=./models
```

Load environment variables:
```bash
source .env
```

### **Config Files**

#### **config.yaml** (Main Configuration)
```yaml
data:
  raw_path: ./data/raw
  processed_path: ./data/processed
  test_size: 0.15
  val_size: 0.15

model:
  sequence_length: 28
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
  batch_size: 32
  epochs: 60
  learning_rate: 0.001

commodities:
  - "Beras Premium"
  - "Beras Medium"
  - "Jagung Pipil Kering"
  - "Kacang Hijau"

modes:
  - "univariate"
  - "multivariate"
```

---

## Running Aplikasi

### **Streamlit App Lokal**

#### **Quick Start**

```bash
# Activate environment
conda activate rapids-24.10

# Run aplikasi
streamlit run streamlit_app/app.py

# Expected output:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
# Network URL: http://192.168.x.x:8501
```

#### **Advanced Options**

```bash
# Run dengan custom port
streamlit run streamlit_app/app.py --server.port 8502

# Run dengan custom config
streamlit run streamlit_app/app.py --logger.level=debug

# Disable cache (untuk development)
streamlit run streamlit_app/app.py --client.caching=false

# Run dengan specific GPU
CUDA_VISIBLE_DEVICES=0 streamlit run streamlit_app/app.py
```

### **Training Model**

```bash
# Train all models (univariate + multivariate)
python src/train.py

# Train specific commodity
python src/train.py --commodity "Beras Medium"

# Train specific mode
python src/train.py --mode univariate

# With custom hyperparameters
python src/train.py \
  --epochs 100 \
  --batch_size 64 \
  --learning_rate 0.0005

# GPU selection
CUDA_VISIBLE_DEVICES=0 python src/train.py
```

### **Running Tests**

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_data_processor.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Generate coverage report
open htmlcov/index.html
```

### **Data Preparation**

```bash
# Prepare data untuk training
python src/data/prepare.py

# Dengan custom paths
python src/data/prepare.py \
  --raw-path ./data/raw \
  --output-path ./data/processed

# Verbose output
python src/data/prepare.py --verbose
```

---

## Deployment ke Cloud

### **Streamlit Cloud** (Recommended untuk quick deployment)

#### **Prerequisites**
- GitHub account dengan repo
- Streamlit account (login dengan GitHub)

#### **Step 1: Prepare Repository**

```bash
# Ensure all files committed
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main

# Required files:
# - streamlit_app/app.py
# - streamlit_app/pages/2_Prediksi_Lanjutan.py
# - streamlit_app/utils/
# - models/*.pth
# - requirements.txt
```

#### **Step 2: Create requirements.txt**

Minimal requirements untuk Streamlit Cloud:

```
streamlit==1.28.0
pandas==2.0.0
numpy==1.24.0
torch==2.0.0
plotly==5.17.0
scikit-learn==1.3.0
pyyaml==6.0
```

**Note**: cuDF (RAPIDS) tidak bisa di-host di Streamlit Cloud (memerlukan GPU). Gunakan standard pandas.

Create `streamlit_app/requirements.txt`:
```bash
cp requirements.txt streamlit_app/requirements.txt

# Edit untuk remove RAPIDS
nano streamlit_app/requirements.txt
# Remove: cudf, rapids, cuda lines
```

#### **Step 3: Deploy to Streamlit Cloud**

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Connect GitHub:
   - Repository: your-username/dsproject_agriculturepredict
   - Branch: main
   - Main file path: streamlit_app/app.py
4. Click "Deploy"

#### **Step 4: Configure Secrets**

In Streamlit Cloud dashboard:
1. Go to App settings → Secrets
2. Add any API keys atau credentials (jika ada)
3. Streamlit restart otomatis dengan secrets

#### **Step 5: Monitor Deployment**

```bash
# View logs
streamlit logs YOUR_APP_ID

# App URL format:
# https://app-username-projectname.streamlit.app
```

### **Docker Deployment** (Untuk production)

#### **Create Dockerfile**

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python & dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git

# Copy files
COPY requirements.txt .
COPY streamlit_app/ ./streamlit_app/
COPY models/ ./models/
COPY data/ ./data/

# Install Python packages
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### **Build & Run Docker Image**

```bash
# Build image
docker build -t agriculture-predict:latest .

# Run container
docker run -p 8501:8501 \
  --gpus all \
  agriculture-predict:latest

# Run dengan GPU selection
docker run -p 8501:8501 \
  --gpus 'device=0' \
  agriculture-predict:latest

# Access di http://localhost:8501
```

#### **Push to Docker Hub** (Optional)

```bash
docker login
docker tag agriculture-predict:latest username/agriculture-predict:latest
docker push username/agriculture-predict:latest
```

### **Heroku Deployment** (Legacy, not recommended)

```bash
# Install Heroku CLI
curl https://cli.heroku.com/install.sh | sh

# Login
heroku login

# Create app
heroku create your-app-name

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

---

## Troubleshooting

### **GPU Issues**

#### **CUDA Not Available**

```bash
# Check CUDA installation
nvidia-smi

# Output should show GPU info
# If error: Install NVIDIA drivers first

# Verify PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"

# Solution:
# 1. Install NVIDIA drivers: https://developer.nvidia.com/cuda-downloads
# 2. Reinstall PyTorch: conda install pytorch::pytorch pytorch::pytorch-cuda=11.8 -c pytorch
```

#### **Out of Memory Error**

```
RuntimeError: CUDA out of memory

Solution:
1. Reduce batch_size: --batch_size 16 (instead of 32)
2. Reduce model size: --hidden_size 32 (instead of 64)
3. Use CPU: unset CUDA_VISIBLE_DEVICES
4. Monitor VRAM: nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -lms 500
```

### **Environment Issues**

#### **Import Error: cudf not found**

```
ModuleNotFoundError: No module named 'cudf'

Solution 1: Install RAPIDS
conda install -c nvidia -c conda-forge cudf=24.10 -y

Solution 2: Use CPU version (slower)
Remove import cudf from code, use pandas instead
```

#### **PyTorch CUDA Mismatch**

```
RuntimeError: CUDA runtime error

Solution:
1. Check CUDA version: nvidia-smi
2. Reinstall PyTorch matching CUDA:
   conda install pytorch::pytorch pytorch::pytorch-cuda=11.8 -c pytorch -y
```

### **Application Issues**

#### **Streamlit App Won't Load**

```bash
# Check if port is in use
lsof -i :8501

# Kill process if needed
kill -9 <PID>

# Run dengan different port
streamlit run streamlit_app/app.py --server.port 8502
```

#### **Models Not Found**

```
FileNotFoundError: models/lstm_Beras_Premium.pth

Solution:
1. Verify model files exist: ls models/*.pth
2. Check working directory: pwd
3. Run dari project root directory
```

#### **Data Loading Error**

```
FileNotFoundError: data/raw/DATASET HARGA KOMODITI.csv

Solution:
1. Verify data files: ls data/raw/
2. Check file names (case-sensitive)
3. Run data preparation: python src/data/prepare.py
```

### **Performance Issues**

#### **Slow Inference**

```bash
# Profile inference speed
python -c "
import torch
import time
from streamlit_app.utils.model_loader import load_model

model = load_model('Beras Premium')
x = torch.randn(1, 28, 7).cuda()

start = time.time()
for _ in range(100):
    _ = model(x)
end = time.time()

print(f'Avg inference time: {(end-start)/100*1000:.2f}ms')
"

# Solutions:
1. Use CPU jika GPU bandwidth bottleneck
2. Reduce sequence length (current: 28 days)
3. Use ONNX runtime untuk faster inference
```

#### **High Memory Usage**

```bash
# Monitor memory
watch -n 1 nvidia-smi

# Solutions:
1. Close other applications
2. Reduce batch size untuk training
3. Run pada different GPU: CUDA_VISIBLE_DEVICES=1
```

---

## Verification Checklist

```bash
□ Python 3.10+ installed
□ conda environment created
□ PyTorch GPU available
□ RAPIDS cuDF installed
□ Streamlit runs locally
□ All models downloaded
□ Data files present
□ No import errors
□ Test prediction works
□ UI loads without errors
```

---

**Document Version**: 1.0  
**Last Updated**: June 2026  
**Tested Environments**: Ubuntu 22.04 LTS, WSL2, macOS 12+

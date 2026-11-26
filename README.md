# facelog-model

Repositori kolaborasi **Sains Data**, **Informatika Medis**, dan **Informatika** untuk riset, *training*, evaluasi, dan *packaging* model pengenalan wajah (*face recognition*) pada proyek **Facelog**.

![Podman](https://img.shields.io/badge/Podman-892CA0?style=for-the-badge&logo=podman&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)

## Struktur Folder

```text
facelog-model/
├── .agents/                 # Blueprint, rules, dan dokumentasi agen
├── configs/                 # Konfigurasi data & model (YAML)
├── dataset/                 # Data mentah, hasil preprocessing, dan daftar split
│   ├── raw/                 # Data asli hasil penarikan (di-ignore Git)
│   ├── processed/           # Hasil deteksi, crop, dan alignment (di-ignore Git)
│   └── splits/              # Daftar split train/val/test (di-track Git)
├── models/                  # Bobot model
│   ├── checkpoints/         # Checkpoint selama training (di-ignore Git)
│   └── exported/            # Model final siap integrasi (.onnx/.tflite, di-ignore Git)
├── notebook/                # Notebook eksplorasi & eksperimen
├── scripts/                 # Script CLI (contoh: pull_data.py)
├── serving/                 # (Opsional) API inferensi untuk tim aplikasi
├── src/                     # Kode Python inti yang modular
│   ├── data/                # Penarikan data & preprocessing
│   ├── models/              # Wrapper detektor & face embedder
│   ├── training/            # Pipeline training / fine-tuning
│   ├── evaluation/          # Metrik evaluasi & benchmark
│   └── inference/           # Pipeline inferensi end-to-end
├── tests/                   # Unit test & sanity check
├── .env.example             # Template kredensial environment
├── .gitignore
└── requirements.txt
```

## Setup Environment

1. Buat dan aktifkan virtual environment dengan **Python 3.11** (Windows PowerShell):
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Pasang dependensi pengembangan lokal:
   ```powershell
   pip install -r requirements-dev.txt
   ```
3. Salin template environment jika butuh kredensial Kaggle atau sumber data lain:
   ```powershell
   copy .env.example .env
   ```

## Alur Kerja Data

1. Verifikasi konfigurasi penarikan dataset (*dry-run*):
   ```powershell
   python scripts/pull_data.py --dataset lfw --dry-run
   ```
2. Tarik dataset Kaggle ke `dataset/raw/<nama_dataset>/` (`lfw`, `pins`, `vggface2`, atau `all`):
   ```powershell
   python scripts/pull_data.py --dataset lfw
   python scripts/pull_data.py --dataset pins
   python scripts/pull_data.py --dataset vggface2
   ```
3. Proses data mentah dari `dataset/raw/` ke `dataset/processed/` menggunakan `src/data/preprocess.py`.

## Menjalankan dengan Docker (Serving API)

Docker disiapkan khusus untuk menjalankan hasil akhir model (*inference/serving API* di port `8000`):

1. Build dan jalankan container:
   ```bash
   docker compose up --build -d
   ```
2. Cek status service dan kesiapan model:
   ```bash
   curl http://localhost:8000/health
   ```

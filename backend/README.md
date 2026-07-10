# FastAPI ML Backend Service

This is the Python-based backend service built with FastAPI, designed to serve Machine Learning predictions, handle data processing, and support model inference.

## Getting Started

### 1. Prerequisites
- Python 3.10+ installed.

### 2. Set Up a Virtual Environment
From the `/backend` directory:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Development Server
```bash
uvicorn app.main:app --reload --port 8000
```
The documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

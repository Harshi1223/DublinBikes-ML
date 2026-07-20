# 🚲 Dublin Bikes Station Availability Prediction System

A machine learning-powered web application that predicts Dublin Bikes station availability using historical data and displays real-time station information from the JCDecaux API.

---

## 📌 Project Overview

This project predicts the availability of bikes and bike stands at Dublin Bikes stations using multiple machine learning algorithms. The system integrates historical station data with live station information to provide accurate predictions and real-time monitoring.

The application consists of:

- FastAPI Backend
- Streamlit Frontend
- Machine Learning Models
- JCDecaux Live API
- AWS EC2 Deployment

---

# System Architecture

```
                +--------------------+
                | Streamlit Frontend |
                +---------+----------+
                          |
                    REST API Calls
                          |
                +---------v----------+
                |   FastAPI Backend  |
                +---------+----------+
                          |
          +---------------+----------------+
          |                                |
          |                                |
+---------v---------+          +-----------v-----------+
| Machine Learning  |          | JCDecaux Live API    |
| Models            |          | Real-time Bike Data  |
+-------------------+          +-----------------------+
```

---

# Features

- Predict bike availability
- Predict available bike stands
- Multiple ML models
- Live station information
- Real-time bike availability
- Interactive Streamlit dashboard
- REST API
- AWS Deployment

---

# Project Structure

```
DublinBikes-ML/
│
├── app.py                     # Streamlit frontend
├── main.py                    # FastAPI backend
├── requirements.txt
├── Dockerfile
├── .gitignore
├── README.md
│
├── models/
│   ├── xgboost/
│   ├── random_forest/
│   ├── logistic_regression/
│   └── lstm/
│
├── data/
│   ├── station_names.csv
│   └── ...
│
├── static/
│
├── utils/
│
├── notebooks/
│
└── screenshots/
```

---

# Technologies Used

## Programming

- Python 3.12
- FastAPI
- Streamlit

## Machine Learning

- XGBoost
- Random Forest
- Logistic Regression
- LSTM

## Libraries

- Pandas
- NumPy
- Scikit-learn
- TensorFlow/Keras
- Requests
- Joblib

## Cloud

- AWS EC2 (Ubuntu 24.04)
- Nginx (optional)
- Systemd Services

---

# Machine Learning Models

The project supports four prediction models.

| Model | Purpose |
|---------|---------|
| XGBoost | Primary prediction model |
| Random Forest | Ensemble prediction |
| Logistic Regression | Binary classification |
| LSTM | Time-series prediction |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Harshi1223/DublinBikes-ML.git

cd DublinBikes-ML
```

---

## Create Virtual Environment

```bash
python3 -m venv .venv
```

Activate

Linux

```bash
source .venv/bin/activate
```

Windows

```powershell
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file.

```text
JCDECAUX_API_KEY=YOUR_API_KEY
```

---

# Running the Backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Backend

```
http://localhost:8000
```

Swagger

```
http://localhost:8000/docs
```

---

# Running the Frontend

```bash
streamlit run app.py
```

Frontend

```
http://localhost:8501
```

---

# API Endpoints

| Endpoint | Description |
|------------|------------|
| / | Health Check |
| /predict | Predict station availability |
| /stations | List all stations |
| /models | Available ML models |
| /health | API Health |

---

# AWS Deployment

## Launch EC2

Recommended

- Ubuntu 24.04 LTS

---

## Install Packages

```bash
sudo apt update

sudo apt install python3-pip python3-venv git
```

---

## Clone Repository

```bash
git clone https://github.com/Harshi1223/DublinBikes-ML.git
```

---

## Install Dependencies

```bash
python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

---

## Run FastAPI

```bash
uvicorn main:app \
--host 0.0.0.0 \
--port 8000
```

---

## Run Streamlit

```bash
streamlit run app.py \
--server.port 8501 \
--server.address 0.0.0.0
```

---

# Deploy as Systemd Services

FastAPI

```bash
sudo systemctl start fastapi

sudo systemctl enable fastapi
```

Check status

```bash
sudo systemctl status fastapi
```

Restart

```bash
sudo systemctl restart fastapi
```

---

Streamlit

```bash
sudo systemctl start streamlit

sudo systemctl enable streamlit
```

Restart

```bash
sudo systemctl restart streamlit
```

---

View logs

```bash
journalctl -u fastapi -f

journalctl -u streamlit -f
```

---

# Project Workflow

```
User
   │
   ▼
Streamlit
   │
   ▼
FastAPI
   │
   ├────────► Load ML Model
   │
   ├────────► Historical Dataset
   │
   └────────► JCDecaux Live API
                   │
                   ▼
            Live Bike Availability
                   │
                   ▼
              Prediction Response
                   │
                   ▼
               Streamlit UI
```

---

# Known Limitation

The historical dataset contains Station 30 (**PARNELL SQUARE NORTH**), which is no longer available in the current JCDecaux live API. Predictions remain available for this station using historical data; however, live bike availability cannot be displayed because the station is absent from the current live feed.

---



---

# Authors

Harshitha S R

MSc Cloud Computing

National College of Ireland

---

# License

This project is developed for academic and research purposes.

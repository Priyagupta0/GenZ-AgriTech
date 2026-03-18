# 🌾 GenZ AgriTech

> **An Intelligent Agricultural Platform Using Deep Learning and Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![Django](https://img.shields.io/badge/Django-4.x-green?logo=django)](https://djangoproject.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet_B0-orange?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🌿 About

GenZ AgriTech is a comprehensive, production-ready AI-driven agricultural platform that addresses the critical challenges faced by farmers — from plant disease outbreaks and weather uncertainty to soil degradation and limited access to government welfare schemes.

The platform integrates **five fully operational AI/ML modules** into a single, intuitive, mobile-responsive interface, making precision agriculture accessible to every farmer regardless of location, literacy level, or resources.

---

## ✨ Features

### 🤖 1. 24/7 AI Chatbot Support
- Powered by **Google Gemini GPT-2.5-flash-pro**
- Intent classification for crop names, diseases, locations, and seasons
- Average response latency: **< 2 seconds**

### 🌿 2. Plant Disease Detection
- **EfficientNet_B0** Convolutional Neural Network
- Trained on **PlantVillage dataset** — 54,305 images, 38 disease classes, 14 crop species
- **99.17% training accuracy | 98.63% test accuracy**
- Drag-and-drop JPEG/PNG upload interface
- Returns: disease name, confidence score, symptoms, and treatment recommendations

### 🪨 3. Soil Type Detection
- ML-based classification: Dataset consists of 7 soil categories
- Input: Drag-and-drop JPEG/PNG upload interface
- Output: comprehensive Soil classification with Crop recommendations
- Accuracy: **97.0%**

### 🌦️ 4. Real-Time Weather Forecasting
- Integrated with **OpenWeatherMap API**
- Geolocation-based or manual location input
- Provides: Temperature, Humidity, Precipitation, Wind Speed, **AQI PM2.5**

### 🏛️ 5. Government Schemes & Guidance Portal
- Aggregates **100+ active central and state government schemes** in real time
- Source: agricoop.gov.in , pib.in and state agriculture department portals
- Each scheme: eligibility criteria, application procedure, documents required, benefits
- Can take print of government schemes page.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3, JavaScript, Tailwind CSS, Font Awesome |
| **Backend** | Python 3.10, Django 4.x |
| **ML/AI** | PyTorch, Scikit-learn |
| **API Used** | Weather.com API, Google Gemini API |

---

## ⚙️ Installation

### Prerequisites

- Python **3.10+**
- pip
- CUDA-capable GPU (recommended for disease detection inference)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Priyagupta0/GenZ-AgriTech
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

fill your API key in dashboard/views:

```
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key
```

### 4. Apply Database Migrations

```bash
python manage.py migrate
```

### 5. Run the Development Server

```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000` in your browser.

---

## 📊 Model Performance

| Module | Metric | Value |
|--------|--------|-------|
| Disease Detection | Training Accuracy | **99.17%** |
| Disease Detection | Test Accuracy | **98.63%** |
| Soil Classification | Accuracy | **97.0%** |
| Chatbot | Avg Response Latency | **< 2.0s** |

---

## 🔭 Future Scope

- [ ] **Yield Prediction Model**
- [ ] **Crop Recommendation Model**
- [ ] **Real-time image capturing through Computer-Vision**

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ for Indian farmers | GenZ AgriTech © 2026</p>

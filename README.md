# Spotify Tracks Popularity Prediction

## Deskripsi
Project ini menggunakan Machine Learning untuk memprediksi popularitas lagu Spotify menggunakan algoritma regresi. Project mencakup pengolahan data, eksplorasi data, pelatihan model, evaluasi, deployment API, dan monitoring sederhana.

## Dataset
Dataset yang digunakan adalah Spotify Tracks Dataset.

Sumber dataset: https://www.kaggle.com/datasets/gauthamvijayaraj/spotify-tracks-dataset-updated-every-week

## Model
Model yang digunakan adalah Gradient Boosting Regressor. Model disimpan dalam format `joblib` dan `pickle`.

## Deployment
Deployment dilakukan menggunakan Flask API. API menerima data dalam format JSON dan memberikan hasil prediksi melalui endpoint `/predict`.

## Cara Menjalankan
1. Install library yang dibutuhkan:
   `pip install flask joblib scikit-learn pandas numpy matplotlib seaborn`

2. Pastikan file model `gbr_model.joblib` dan `testing_deploy.py` berada dalam folder yang sama.

3. Jalankan API:
   `python testing_deploy.py`

4. Buka terminal lain, lalu jalankan pengujian:
   `curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @data.json`

## Monitoring
Monitoring dasar mencakup waktu respons API (latency), error, serta pembahasan metrik evaluasi regresi dan data drift. Monitoring otomatis menggunakan tools khusus belum diterapkan dalam project ini.
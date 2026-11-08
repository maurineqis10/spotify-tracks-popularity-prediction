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


Qisky Maurine Pratiwi
241011400762
05TPLP019
Teknik Informatika, Universitas Pamulang

# Landasan Teori: Metrik Evaluasi Regresi (MAE, MSE, dan R²) pada Prediksi Popularitas Lagu Spotify

Landasan teori ini membahas tiga metrik yang digunakan untuk mengevaluasi model regresi, yaitu Mean Absolute Error (MAE), Mean Squared Error (MSE), dan Koefisien Determinasi (R²). Ketiga metrik tersebut digunakan untuk mengetahui seberapa baik model Machine Learning dalam memprediksi popularitas lagu berdasarkan karakteristik audio pada Spotify Tracks Dataset.

## 2.1 Evaluasi Model Regresi
Evaluasi model regresi merupakan proses untuk mengetahui seberapa baik model dalam memprediksi suatu nilai berdasarkan data yang tersedia. Hasil prediksi akan dibandingkan dengan nilai sebenarnya untuk mengetahui besar kesalahan yang dihasilkan oleh model. Beberapa metrik yang dapat digunakan dalam evaluasi regresi adalah MAE, MSE, RMSE, dan R² (Wahyuzi et al., 2024).

Dalam penelitian prediksi, metrik evaluasi membantu mengetahui apakah model sudah menghasilkan prediksi yang mendekati nilai sebenarnya. Penggunaan MAE, MSE, dan R² juga diterapkan dalam penelitian Putri et al. (2025) untuk mengetahui hasil prediksi menggunakan model Machine Learning.

Pada penelitian ini, evaluasi regresi digunakan untuk mengukur kemampuan algoritma Gradient Boosting Regressor dalam memprediksi popularitas lagu Spotify berdasarkan karakteristik audio lagu.

## 2.2 Mean Absolute Error (MAE)
Mean Absolute Error (MAE) merupakan metode evaluasi yang digunakan untuk menghitung rata-rata selisih antara nilai sebenarnya dengan nilai hasil prediksi. Perhitungan ini menggunakan selisih absolut sehingga kesalahan tidak memperhatikan apakah hasil prediksi lebih tinggi atau lebih rendah dari nilai sebenarnya (Wahyuzi et al., 2024).

Semakin kecil nilai MAE, semakin dekat hasil prediksi dengan nilai sebenarnya. Sebaliknya, jika nilai MAE semakin besar, berarti terdapat kesalahan prediksi yang lebih besar.

Rumus MAE:
$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|$$

Keterangan:
- MAE = rata-rata kesalahan prediksi.
- $n$ = jumlah data.
- $y_i$ = nilai sebenarnya.
- $\hat{y}_i$ = nilai hasil prediksi.

Pada dataset Spotify, MAE digunakan untuk mengetahui rata-rata selisih antara popularitas lagu yang sebenarnya dengan popularitas lagu yang diprediksi oleh model (Setiawan et al., 2024).

## 2.3 Mean Squared Error (MSE)
Mean Squared Error (MSE) merupakan metode evaluasi yang digunakan untuk menghitung rata-rata kesalahan prediksi dengan cara mengkuadratkan selisih antara nilai sebenarnya dan nilai prediksi. Berbeda dengan MAE, MSE memberikan pengaruh lebih besar terhadap kesalahan prediksi yang nilainya tinggi karena selisih tersebut dikuadratkan (Silalahi et al., 2023).

Nilai MSE yang semakin kecil menunjukkan bahwa hasil prediksi semakin mendekati nilai sebenarnya. Oleh karena itu, MSE dapat digunakan untuk mengetahui seberapa besar kesalahan yang dihasilkan oleh model regresi.

Rumus MSE:
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2$$

Keterangan:
- MSE = rata-rata kesalahan kuadrat.
- $n$ = jumlah data.
- $y_i$ = nilai sebenarnya.
- $\hat{y}_i$ = nilai hasil prediksi.

Dalam penelitian ini, MSE digunakan untuk mengetahui besar kesalahan model Gradient Boosting Regressor ketika memprediksi popularitas lagu Spotify. Nilai kesalahan yang lebih kecil menunjukkan hasil prediksi yang lebih baik.

## 2.4 Koefisien Determinasi (R²)
Koefisien Determinasi atau R² merupakan metrik yang digunakan untuk mengetahui seberapa baik model dalam menjelaskan perubahan atau variasi pada data. Nilai R² yang semakin mendekati 1 menunjukkan bahwa model mampu menjelaskan variasi data dengan lebih baik (Putri et al., 2025).

Sebaliknya, nilai R² yang rendah menunjukkan bahwa kemampuan model dalam menjelaskan variasi data masih kurang baik. Nilai R² juga dapat bernilai negatif apabila hasil model lebih buruk dibandingkan penggunaan nilai rata-rata sebagai prediksi.

Rumus R²:
$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}$

Keterangan:
- $R^2$ = koefisien determinasi.
- $y_i$ = nilai sebenarnya.
- $\hat{y}_i$ = nilai hasil prediksi.
- $\bar{y}$ = rata-rata nilai sebenarnya.
- $n$ = jumlah data.

Sebagai contoh, apabila nilai R² sebesar 0,80, berarti model dapat menjelaskan sekitar 80% variasi data. Sementara itu, sisanya belum dapat dijelaskan oleh model. Penggunaan R² sebagai salah satu ukuran evaluasi juga diterapkan dalam penelitian Puspitasari et al. (2023).

## 2.5 Ringkasan Interpretasi
- **MAE (Mean Absolute Error)**
  - Yang diukur: rata-rata selisih antara nilai sebenarnya dan hasil prediksi.
  - Model baik jika: nilai semakin kecil dan mendekati nol.
  - Sumber: Wahyuzi et al. (2024).
- **MSE (Mean Squared Error)**
  - Yang diukur: rata-rata kesalahan prediksi yang dikuadratkan.
  - Model baik jika: nilai semakin kecil dan mendekati nol.
  - Sumber: Silalahi et al. (2023).
- **R² (Koefisien Determinasi)**
  - Yang diukur: kemampuan model dalam menjelaskan variasi data.
  - Model baik jika: nilai semakin mendekati 1.
  - Sumber: Putri et al. (2025).

## Daftar Pustaka
1. Silalahi, D. K., Rumapea, A. C. M., Priharti, W., & Aprillia, B. S. (2023). Forecasting Photovoltaic Output Power Based on Environmental Parameters Using Artificial Neural Network Methods. Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi), 7(6), 1276–1284. https://doi.org/10.29207/resti.v7i6.5214
2. Setiawan, A., Utami, E., & Ariatmanto, D. (2024). Cattle Weight Estimation Using Linear Regression and Random Forest Regressor. Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi), 8(1), 72–79. https://doi.org/10.29207/resti.v8i1.5494
3. Wahyuzi, Z., Luthfi, A., & Fudholi, D. H. (2024). Predicting Smart Office Electricity Consumption in Response to Weather Conditions Using Deep Learning. Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi), 8(1), 100–110. https://doi.org/10.29207/resti.v8i1.5530
4. Putri, S., Fahrudin, T., & Widayanti, A. (2025). Agricultural Cultivation Cost Prediction Using Neural Networks and Feature Importance Analysis. Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi), 9(1), 31–39. https://doi.org/10.29207/resti.v9i1.6003
5. Puspitasari, D. I., Kholdani, A. F. R., Qur’ana, T. W., & Dharmawati, A. (2023). Pemodelan Spasial untuk Analisa Produksi Padi Integrasi Machine Learning. Digital Zone: Jurnal Teknologi Informasi dan Komunikasi, 14(2), 128–137. https://doi.org/10.31849/digitalzone.v14i2.16256

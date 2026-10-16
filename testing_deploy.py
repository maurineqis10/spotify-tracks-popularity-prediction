from flask import Flask, request, jsonify
import joblib
import time

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model yang sudah disimpan
joblib_model = joblib.load("gbr_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["data"]
        mulai = time.time()
        prediction = joblib_model.predict(data)
        waktu = round((time.time() - mulai) * 1000, 2)
        print("Waktu prediksi:", waktu, "ms")
        return jsonify({"prediction": prediction.tolist(), "latency_ms": waktu})
    except Exception as e:
        print("Terjadi error saat prediksi:", e)
        return jsonify({"error": "Data input belum sesuai dengan fitur model."}), 400

if __name__ == "__main__":
    app.run(debug=True)

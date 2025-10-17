from flask import Flask, request, jsonify
import torch
import os

app = Flask(__name__)

# Ruta raíz de prueba
@app.route('/')
def home():
    return jsonify({"status": "ok", "message": "Servidor funcionando correctamente"})


# 🔹 Cargar modelos a demanda
def load_model(model_name):
    model_path = os.path.join("models", f"{model_name}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    return model


# 🔹 Endpoint para detección
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_name = data.get("model")
    image_path = data.get("image_path")

    if not model_name or not image_path:
        return jsonify({"error": "Faltan parámetros: 'model' y 'image_path'"}), 400

    try:
        model = load_model(model_name)
        results = model(image_path)
        detections = results.pandas().xyxy[0].to_dict(orient="records")
        return jsonify({"modelo": model_name, "detecciones": detections})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 🔹 Punto de entrada
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render asigna el puerto automáticamente
    app.run(host='0.0.0.0', port=port)

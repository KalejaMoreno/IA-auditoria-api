import os
import io
import mimetypes
import base64
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO
import openai
import fitz  # PyMuPDF

# ==============================
# CONFIGURACIÓN INICIAL
# ==============================
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="IA Auditoría Odontológica")

# ==============================
# CARGA DE MODELOS YOLO
# ==============================
print("Cargando modelos YOLO...")

yolo_periapical = YOLO("models/periapicales.pt")
yolo_pano_dientes = YOLO("models/panoramicas_dientes.pt")
yolo_pano_diag = YOLO("models/panoramicas_diag.pt")
yolo_fotos = YOLO("models/fotos_dentales.pt")

print("Modelos cargados correctamente ✅")


# ==============================
# FUNCIONES AUXILIARES
# ==============================

def extraer_texto_pdf(pdf_bytes):
    """Extrae texto de un PDF para pasar a GPT."""
    texto = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            texto += page.get_text("text")
    return texto.strip()


def clasificar_con_gpt(file_path, texto_extraido=None):
    """
    Usa GPT-4o para identificar si el archivo o texto corresponde a:
    - historia_clinica
    - rx_panoramica
    - rx_periapical
    - foto_dental
    - otro
    """

    prompt = """
    Eres un experto en odontología y documentación clínica.
    Analiza el contenido del archivo o texto y clasifícalo en una de las siguientes categorías:
    - historia_clinica → si contiene texto, formularios o estructuras típicas de una historia clínica médica u odontológica.
      Palabras comunes: paciente, edad, diagnóstico, tratamiento, anamnesis, odontograma, firma, evolución, motivo de consulta, signos vitales.
    - rx_panoramica → si es una radiografía panorámica dental.
    - rx_periapical → si es una radiografía pequeña de uno o pocos dientes.
    - foto_dental → si es una fotografía intraoral o extraoral.
    - otro → si no corresponde a las anteriores.
    Responde SOLO con el nombre exacto de la categoría.
    """

    mime_type, _ = mimetypes.guess_type(file_path)

    # Si ya tenemos texto (por ejemplo, PDF), enviamos texto a GPT
    if texto_extraido:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en odontología digital."},
                    {"role": "user", "content": prompt + "\n\nTexto del documento:\n" + texto_extraido[:4000]}
                ],
                max_tokens=20
            )
            decision = response.choices[0].message.content.strip().lower()
            return decision
        except Exception as e:
            print(f"❌ Error en GPT (texto): {e}")
            return f"error_gpt: {e}"

    # Si es imagen, enviamos la imagen codificada
    else:
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            file_base64 = base64.b64encode(file_bytes).decode("utf-8")

            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en odontología digital."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{file_base64}"}}
                        ]
                    }
                ],
                max_tokens=20
            )

            decision = response.choices[0].message.content.strip().lower()
            return decision

        except Exception as e:
            print(f"❌ Error en GPT (imagen): {e}")
            return f"error_gpt: {e}"


def clasificar_imagen_yolo(img: Image.Image):
    """Clasifica imágenes dentales con los modelos YOLO."""
    resultados = {
        "periapical": yolo_periapical.predict(img, verbose=False),
        "panoramica": yolo_pano_dientes.predict(img, verbose=False),
        "foto_dental": yolo_fotos.predict(img, verbose=False)
    }

    confs = {}
    for tipo, res in resultados.items():
        if len(res[0].boxes) > 0:
            confs[tipo] = float(res[0].boxes.conf[0])
        else:
            confs[tipo] = 0.0

    mejor_tipo = max(confs, key=confs.get)
    confianza = confs[mejor_tipo]

    if confianza < 0.4:
        return {"categoria": "no_dental", "detalle": None}

    if mejor_tipo == "panoramica":
        res_dientes = yolo_pano_dientes.predict(img, verbose=False)
        res_diag = yolo_pano_diag.predict(img, verbose=False)
        return {
            "categoria": "rx_panoramica",
            "detecciones": {
                "dientes": len(res_dientes[0].boxes),
                "diagnosticos": len(res_diag[0].boxes)
            }
        }

    elif mejor_tipo == "periapical":
        return {"categoria": "rx_periapical", "detecciones": len(resultados["periapical"][0].boxes)}

    elif mejor_tipo == "foto_dental":
        return {"categoria": "foto_dental", "detecciones": len(resultados["foto_dental"][0].boxes)}

    return {"categoria": "no_dental", "detalle": None}


# ==============================
# ENDPOINT PRINCIPAL
# ==============================
@app.post("/procesar/")
async def procesar_archivo(file: UploadFile = File(...)):
    nombre = file.filename
    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", nombre)

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    mime_type, _ = mimetypes.guess_type(temp_path)
    print(f"\n📂 Archivo recibido: {nombre} ({mime_type})")

    # --- Si es PDF: extraer texto y enviar a GPT ---
    if mime_type == "application/pdf":
        with open(temp_path, "rb") as f:
            pdf_bytes = f.read()
        texto = extraer_texto_pdf(pdf_bytes)
        decision = clasificar_con_gpt(temp_path, texto_extraido=texto)
        print(f"GPT dice (PDF): {decision}")

        if "historia" in decision:
            return {"archivo": nombre, "clasificacion": "historia_clinica"}
        else:
            return {"archivo": nombre, "clasificacion": "otro_documento"}

    # --- Si es imagen: primero GPT, luego YOLO si no es historia ---
    elif mime_type and mime_type.startswith("image/"):
        decision = clasificar_con_gpt(temp_path)
        print(f"GPT dice (imagen): {decision}")

        if "historia" in decision:
            return {"archivo": nombre, "clasificacion": "historia_clinica"}

        img = Image.open(temp_path)
        clasificacion_yolo = clasificar_imagen_yolo(img)
        return {
            "archivo": nombre,
            "clasificacion": clasificacion_yolo["categoria"],
            "detalle": clasificacion_yolo.get("detecciones", None)
        }

    else:
        return {"archivo": nombre, "error": f"Tipo de archivo no soportado: {mime_type}"}


# ==============================
# PRUEBAS LOCALES
# ==============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

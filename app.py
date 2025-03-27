import base64
import ChunksAndEmbeddings
from flask import Flask, request, jsonify

app = Flask(__name__)

# Inicializar el objeto una sola vez para reutilizarlo a nivel de aplicación
MainObject = ChunksAndEmbeddings.ChunksAndEmbeddings()
MainObject.Load_LanguageModel()
print("Modelos cargados y listos para usar")

@app.route('/process', methods=['POST'])
def process_text():
    try:
        # Obtener datos de la petición
        if request.is_json:
            data_json = request.get_json()
            mode = data_json.get("mode", "chunks")  # Por defecto, modo de chunks
            
            # Manejar texto en base64 si está codificado así
            if "textbase64" in data_json:
                text = data_json.get("textbase64", "")
                text = base64.b64decode(text)
                text = text.decode('utf-8')
            else:
                # Si no está en base64, usar el texto directamente
                text = data_json.get("text", "")
        else:
            # Si no es JSON, usar el contenido como texto
            mode = "chunks"
            text = request.data.decode('utf-8')
            
        print(f"   Modo seleccionado [{mode}]")
        print(f"   Texto recibido    [{len(text):,}]")
        
        if mode == "single":
            # Modalidad para texto corto - devolver un solo embedding
            cleaned_text = MainObject.CleanText(text)
            embedding = MainObject.GetSingleEmbedding(cleaned_text)
            response = {
                "mode": "single",
                "text": cleaned_text,
                "embedding": embedding.tolist()
            }
            print(f"   Embedding simple generado")
        else:
            # Modalidad para texto largo - dividir en chunks
            Chunks = MainObject.GetChunks(text)
            Embedding = MainObject.GetEmbeddings(Chunks)
            print(f"   Chunks generados [{len(Chunks):,}]")
            response = {
                "mode": "chunks",
                "chunks": Chunks,
                "embedding": Embedding.tolist()
            }
            
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Ruta para verificar que el servicio está activo
@app.route('/health', methods=['GET'])
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Servicio funcionando correctamente"})

@app.route('/test', methods=['GET'])
def simple_test():
    base64text = request.args.get('base64text')  # Valor por defecto: None 
    if base64text == None:
        return jsonify({"status": "ok", "message": "Sin texto en parámetro 'base64text'"})
    else:
        text         = base64.b64decode(base64text)
        text         = text.decode('utf-8')
        cleaned_text = MainObject.CleanText(text)
        embedding    = MainObject.GetSingleEmbedding(cleaned_text)
        response = {
                "mode": "test",
                "text": cleaned_text,
                "embedding": embedding[0].tolist()
            }
        print(f"   Embedding simple generado")
        return jsonify(response)

if __name__ == '__main__':
    # En desarrollo, usar debug=True
    # Para producción, cambiar a host='0.0.0.0' para aceptar conexiones externas
    app.run(host='0.0.0.0', port=5000)
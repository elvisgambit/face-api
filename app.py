from flask import Flask, request, jsonify
from deepface import DeepFace
import os

app = Flask(__name__)
UPLOAD_FOLDER = "imagens"  # pasta com as fotos cadastradas

@app.route('/verificar', methods=['POST'])
def verificar():
    if 'foto' not in request.files:
        return jsonify({"erro": "Nenhuma foto enviada"}), 400

    foto = request.files['foto']
    path_foto_enviada = os.path.join("temp", foto.filename)
    foto.save(path_foto_enviada)

    melhor_match = None
    menor_distancia = 100

    for filename in os.listdir(UPLOAD_FOLDER):
        caminho_imagem = os.path.join(UPLOAD_FOLDER, filename)

        try:
            resultado = DeepFace.verify(img1_path=path_foto_enviada, img2_path=caminho_imagem, enforce_detection=False)
            if resultado["verified"] and resultado["distance"] < menor_distancia:
                menor_distancia = resultado["distance"]
                melhor_match = filename
        except Exception as e:
            continue

    if melhor_match:
        id_usuario = os.path.splitext(melhor_match)[0]  # Nome do arquivo sem extensão
        return jsonify({
            "id": id_usuario,
            "distancia": menor_distancia
        })
    else:
        return jsonify({"erro": "Nenhum rosto correspondente encontrado"}), 404

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

from flask import Flask, request, jsonify
import cv2
import face_recognition
import os
import numpy as np
import uuid
import base64
import requests
from io import BytesIO
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
KNOWN_FACES_DIR = 'known_faces'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Variáveis globais para armazenar os encodings e face_keys
known_face_encodings = []
known_face_keys = []


def load_known_faces():
    """Carrega todas as imagens da pasta known_faces e cria seus encodings."""
    global known_face_encodings, known_face_keys
    known_face_encodings.clear()
    known_face_keys.clear()

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.startswith('.'):
            continue  # Ignorar arquivos ocultos

        face_key = os.path.splitext(filename)[0]
        image_path = os.path.join(KNOWN_FACES_DIR, filename)

        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_keys.append(face_key)
            else:
                print(f"Nenhum rosto encontrado na imagem {filename}. Pulando...")
        except Exception as e:
            print(f"Erro ao processar {filename}: {str(e)}")


# Carrega os rostos conhecidos na inicialização
load_known_faces()


def convert_image_to_array(img_data, img_type):
    """
    Converte dados de imagem nos formatos file/base64/url para array RGB.
    """
    try:
        if img_type == 'file':
            if not isinstance(img_data, bytes):
                return None
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        elif img_type == 'base64':
            header, encoded = img_data.split(",", 1) if "," in img_data else ("", img_data)
            data = base64.b64decode(encoded)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

        elif img_type == 'url':
            response = requests.get(img_data, timeout=10)
            img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

        else:
            raise ValueError("Tipo de imagem inválido")

        if img is None:
            raise ValueError("Erro ao carregar imagem")

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    except Exception as e:
        raise RuntimeError(f"Erro ao processar imagem: {str(e)}")


@app.route('/envia', methods=['POST'])
def upload_image():
    content_type = request.content_type

    try:
        if content_type == 'application/json':
            json_data = request.get_json()
            img_type = json_data.get('type')
            data = json_data.get('data')

            if not img_type or data is None:
                return jsonify({'error': 'Dados incompletos'}), 400

            img_array = convert_image_to_array(data, img_type)

        elif content_type.startswith('multipart/form-data'):
            img_type = request.form.get('type')
            data = request.form.get('data')

            if img_type == 'file':
                if 'data' not in request.files:
                    return jsonify({'error': 'Arquivo não fornecido'}), 400
                file = request.files['data']
                if file.filename == '':
                    return jsonify({'error': 'Nome do arquivo vazio'}), 400
                img_bytes = file.read()
                img_array = convert_image_to_array(img_bytes, 'file')
            else:
                if not img_type or data is None:
                    return jsonify({'error': 'Dados incompletos'}), 400
                img_array = convert_image_to_array(data, img_type)
        else:
            return jsonify({'error': 'Tipo de conteúdo não suportado'}), 415

        # Detecta rosto
        encodings = face_recognition.face_encodings(img_array)
        if not encodings:
            return jsonify({'error': 'Nenhum rosto encontrado na imagem'}), 400

        # Gera face_key
        face_key = str(uuid.uuid4())
        img_pil = Image.fromarray(img_array)
        img_path = os.path.join(KNOWN_FACES_DIR, f"{face_key}.jpg")
        img_pil.save(img_path, format="JPEG")

        # Recarrega faces
        load_known_faces()

        return jsonify({
            'face_key': face_key,
            'message': 'Imagem salva com sucesso'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/reconhece', methods=['POST'])
def recognize_face():
    content_type = request.content_type

    try:
        if content_type == 'application/json':
            json_data = request.get_json()
            img_type = json_data.get('type')
            data = json_data.get('data')

            if not img_type or data is None:
                return jsonify({'error': 'Dados incompletos'}), 400

            unknown_image = convert_image_to_array(data, img_type)

        elif content_type.startswith('multipart/form-data'):
            img_type = request.form.get('type')
            data = request.form.get('data')

            if img_type == 'file':
                if 'data' not in request.files:
                    return jsonify({'error': 'Arquivo não fornecido'}), 400
                file = request.files['data']
                if file.filename == '':
                    return jsonify({'error': 'Nome do arquivo vazio'}), 400
                img_bytes = file.read()
                unknown_image = convert_image_to_array(img_bytes, 'file')
            else:
                if not img_type or data is None:
                    return jsonify({'error': 'Dados incompletos'}), 400
                unknown_image = convert_image_to_array(data, img_type)
        else:
            return jsonify({'error': 'Tipo de conteúdo não suportado'}), 415

        # Detecta rosto
        unknown_encodings = face_recognition.face_encodings(unknown_image)
        if not unknowan_encodings:
            return jsonify({
                "match": False,
                "face_key": None,
                "distance": None,
                "message": "Nenhum rosto encontrado na imagem"
            }), 200

        unknown_encoding = unknown_encodings[0]

        matches = face_recognition.compare_faces(known_face_encodings, unknown_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            face_key = known_face_keys[best_match_index]
            distance = float(face_distances[best_match_index])
            return jsonify({
                'match': True,
                'face_key': face_key,
                'distance': round(distance, 2),
                'message': f'Rosto identificado com face_key: {face_key}'
            }), 200
        else:
            return jsonify({
                'match': False,
                'face_key': None,
                'distance': None,
                'message': 'Rosto desconhecido'
            }), 200

    except Exception as e:
        return jsonify({
            'match': False,
            'face_key': None,
            'distance': None,
            'error': str(e),
            'message': 'Erro ao processar imagem'
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

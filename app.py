import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import nltk
import requests
from mlxtend.frequent_patterns import apriori, association_rules

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def recomendar_servicios(rules, servicios_usuario_unicos):
    reglas_usuario = rules[rules['antecedents'].apply(lambda x: any(item in x for item in servicios_usuario_unicos))]
    recomendaciones = set()
    for index, row in reglas_usuario.iterrows():
        for item in row['consequents']:
            if item not in servicios_usuario_unicos:
                recomendaciones.add(item)
    
    
    return list(recomendaciones)

# Inicializar lematizador y stopwords
lemmatizer = WordNetLemmatizer()
stop_words_en = set(stopwords.words('english'))
# Función de preprocesamiento
def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words_en and word.isalpha()]
    return ' '.join(tokens)


# Cargar el modelo y el vectorizador
MLP_model = joblib.load('MLP_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
rules = joblib.load('association_rules.pkl')

# Crear la aplicación Flask
app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

@app.route('/predict/comment', methods=['POST'])
def predict_comment():
    #Recibe el comentario
    data = request.get_json(force=True)
    comment = data.get('comment', '')

    # "Verifica si se paso el comentario"
    if not comment:
        return jsonify({'error': 'No comment provided'}), 400

    # Preprocesar el comentario, ES DECIR QUITA PALABRAS COMO THE, OF AND, ETC.
    preprocessed_comment = preprocess(comment)

    # Crear un DataFrame con el comentario
    df = pd.DataFrame([preprocessed_comment], columns=['Comentario'])

    # Vectorizar el comentario es decir lo pasa a numero
    X = vectorizer.transform(df['Comentario'])

    # Hacer la predicción
    prediction = MLP_model.predict(X)[0]

    # Convertir la predicción a un tipo de dato entero
    prediction = int(prediction)

    # Retornar el resultado de la predicción, 1 para bueno, 0 para malo, 2 para neutro
    return jsonify({'prediction': prediction})

@app.route('/predict/services', methods=['POST'])
def predict_services():
    data = request.get_json(force=True)
    token = data.get('user', '')
    url = 'https://db-api-mygarden-llc.onrender.com/api/schedule/userservices'
    headers = {
        'Authorization': 'Bearer ' + token,
        'rol': 'client'
    }

    # Hacer la petición GET
    response = requests.get(url, headers=headers)

    # Verificar el estado de la respuesta
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            services = data['services']
            
            # Extraer los IDs de los servicios
            service_ids = [service['service']['_id'] for service in services]

            # Eliminar IDs duplicados
            unique_service_ids = list(set(service_ids))

            # Imprimir la lista de IDs y su tamaño
            print("Lista de IDs de servicios únicos:", unique_service_ids)
            print("Tamaño de la lista de IDs únicos:", len(unique_service_ids))
        else:
            print("Error en la respuesta del servidor:", data['msg'])
    else:
        print(f"Error en la petición: {response.status_code} - {response.text}")

    recomendaciones_para_u1 = recomendar_servicios(rules, unique_service_ids)

    return jsonify({'recommendations': recomendaciones_para_u1})

if __name__ == '__main__':
    app.run(debug=True)

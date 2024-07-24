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

# Inicializar lematizador y stopwords
lemmatizer = WordNetLemmatizer()
stop_words_en = set(stopwords.words('english'))
df_completo = pd.DataFrame()  # Inicializar como DataFrame vacío
df_usuarios = pd.DataFrame()  # Inicializar como DataFrame vacío

# Función de preprocesamiento
def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words_en and word.isalpha()]
    return ' '.join(tokens)

def load_data():
    global df_completo, df_usuarios
    url = 'http://localhost:5000/api/services/get'
    response = requests.get(url)

    # Verifica que la respuesta sea exitosa
    if response.status_code == 200:
        data = response.json()
        if data["success"]:
            services = data["services"]
            
            # Lista para almacenar los datos
            data_list = []

            # Extrae los datos necesarios
            for service in services:
                _id = service["_id"]
                service_name = service["name"]
                description = service["description"]
                typeService = service["tipoDeServicio"]["_id"]

                # Agrega los datos a la lista
                data_list.append([_id, service_name, description, typeService])

            # Crea el DataFrame
            df_servicios = pd.DataFrame(data_list, columns=["_id", "service", "description", "typeService"])
        else:
            print("Error: No se pudo obtener la lista de servicios.")
    else:
        print(f"Error: La petición falló con el código de estado {response.status_code}.")

    urlScheduledServices = 'http://localhost:5000/api/schedule/getServices'
    response = requests.get(urlScheduledServices)

    # Verifica que la respuesta sea exitosa
    if response.status_code == 200:
        data = response.json()
        if data["success"]:
            services = data["services"]
            
            # Lista para almacenar los datos
            data_list = []

            # Extrae los datos necesarios
            for service_data in services:  # Cambia el nombre de la variable service a service_data
                user = service_data["user"]["_id"]
                service_id = service_data["service"]["_id"]
                description = service_data["description"]  # Accede a la descripción del servicio

                # Agrega los datos a la lista
                data_list.append([user, service_id, description])

            # Crea el DataFrame
            df_servicios_agendados = pd.DataFrame(data_list, columns=["user", "service", "description"])
        else:
            print("Error: No se pudo obtener la lista de servicios.")
    else:
        print(f"Error: La petición falló con el código de estado {response.status_code}.")

    urlScheduledServices = 'http://localhost:5000/api/typeservice/get'
    response = requests.get(urlScheduledServices)

    # Verifica que la respuesta sea exitosa
    if response.status_code == 200:
        data = response.json()
        if data["success"]:
            types = data["tipesServices"]
            
            # Lista para almacenar los datos
            data_list = []

            # Extrae los datos necesarios
            for type in types:  # Cambia el nombre de la variable service a service_data
                _id = type["_id"]  # Accede a la descripción del servicio
                tipo = type["tipo"]  # Accede a la descripción del servicio

                # Agrega los datos a la lista
                data_list.append([_id, tipo])

            # Crea el DataFrame
            df_tipos_servicios = pd.DataFrame(data_list, columns=["_id", "tipo"])
        else:
            print("Error: No se pudo obtener la lista de servicios.")
    else:
        print(f"Error: La petición falló con el código de estado {response.status_code}.")

    url = 'http://localhost:5000/api/user/get'
    response = requests.get(url)

    # Verifica que la respuesta sea exitosa
    if response.status_code == 200:
        data = response.json()
        if data["success"]:
            users = data["users"]
            
            # Lista para almacenar los datos
            data_list = []

            # Extrae los datos necesarios
            for user in users:  # Cambia el nombre de la variable service a service_data
                _id = user["_id"]  # Accede a la descripción del servicio
                name = user["name"]  # Accede a la descripción del servicio

                # Agrega los datos a la lista
                data_list.append([_id, name])

            # Crea el DataFrame
            df_usuarios = pd.DataFrame(data_list, columns=["_id", "name"])
        else:
            print("Error: No se pudo obtener la lista de servicios.")
    else:
        print(f"Error: La petición falló con el código de estado {response.status_code}.")

    # Realizar las uniones de los DataFrames
    df_completo = df_servicios_agendados.merge(df_servicios, left_on='service', right_on='_id', suffixes=('_agendado', '_servicio'))

    # Verifica si 'typeService' está presente antes de proceder con la siguiente unión
    if 'typeService' in df_completo.columns:
        df_completo = df_completo.merge(df_tipos_servicios, left_on='typeService', right_on='_id', suffixes=('', '_tipo'))
        
        # Eliminar columnas solo si existen
        if '_id_tipo' in df_completo.columns:
            df_completo = df_completo.drop(columns=['_id_tipo', 'typeService'])
    else:
        print("'typeService' no se encontró en las columnas después de la primera unión.")

    # Verifica si 'user' está presente antes de proceder con la siguiente unión
    if 'user' in df_completo.columns:
        df_completo = df_completo.merge(df_usuarios, left_on='user', right_on='_id', suffixes=('', '_usuario'))
        
        # Eliminar columna solo si existe
        if '_id_usuario' in df_completo.columns:
            df_completo = df_completo.drop(columns=['_id_usuario'])
    else:
        print("'user' no se encontró en las columnas después de la segunda unión.")

    # Eliminar columnas innecesarias
    df_completo.drop(columns=['_id', 'description_servicio', 'name'], inplace=True)

    # Renombrar columnas para mayor claridad
    df_completo.rename(columns={
        'user': 'user',
        'service_agendado': 'service',
        'description_agendado': 'description',
        'service_servicio': 'service_name',
        'tipo': 'service_type',
    }, inplace=True)

def recomendar_servicios(rules, df_completo, usuario):
    servicios_usuario_unicos = set(df_completo[df_completo['user'] == usuario]['service'])
    print(f"servicios que ya agendo {len(servicios_usuario_unicos)}")
    # Filtrar reglas donde los servicios del usuario están en los antecedentes
    reglas_usuario = rules[rules['antecedents'].apply(lambda x: any(item in x for item in servicios_usuario_unicos))]

    recomendaciones = set()
    for index, row in reglas_usuario.iterrows():
        for item in row['consequents']:
            if item not in servicios_usuario_unicos:
                recomendaciones.add(item)

    return list(recomendaciones)

# Inicializar los datos al cargar la aplicación
load_data()

# Cargar el modelo y el vectorizador
MLP_model = joblib.load('MLP_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Crear la aplicación Flask
app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

@app.route('/predict/comment', methods=['POST'])
def predict_comment():
    data = request.get_json(force=True)
    comment = data.get('comment', '')

    if not comment:
        return jsonify({'error': 'No comment provided'}), 400

    # Preprocesar el comentario
    preprocessed_comment = preprocess(comment)

    # Crear un DataFrame con el comentario
    df = pd.DataFrame([preprocessed_comment], columns=['Comentario'])

    # Vectorizar el comentario
    X = vectorizer.transform(df['Comentario'])

    # Hacer la predicción
    prediction = MLP_model.predict(X)[0]

    # Convertir la predicción a un tipo de dato JSON serializable
    prediction = int(prediction)

    # Retornar el resultado de la predicción
    return jsonify({'prediction': prediction})

@app.route('/predict/services', methods=['POST'])
def predict_services():
    data = request.get_json(force=True)
    user = data.get('user', '')

    # Verificar si el usuario está en df_usuarios
    if user not in df_usuarios['_id'].values:
        load_data()  # Recargar los datos si el usuario no está presente

    incidence_matrix = df_completo.pivot_table(index='user', columns='service', values='service_name', aggfunc=pd.Series.nunique, fill_value=0)

    # Convertir la matriz de incidencia a tipo booleano
    incidence_matrix = incidence_matrix.astype(bool)

    frequent_itemsets = apriori(incidence_matrix, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.35)

    recomendaciones_para_u1 = recomendar_servicios(rules, df_completo, user)

    return jsonify({'recommendations': recomendaciones_para_u1})

if __name__ == '__main__':
    app.run(debug=True)

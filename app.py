from flask import Flask, request, jsonify
from flask_mysqldb import MySQL
import pickle
import pandas as pd
from MySQLdb.cursors import DictCursor

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'  # Ganti dengan host Anda
app.config['MYSQL_USER'] = 'root'  # Ganti dengan username Anda
app.config['MYSQL_PASSWORD'] = ''  # Ganti dengan password Anda
app.config['MYSQL_DB'] = 'soundly'  # Ganti dengan nama database Anda
mysql = MySQL(app)

# Memuat model dari file pickle
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Fungsi untuk melakukan prediksi
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

@app.route('/predict', methods=['POST'])
def get_prediction():
    data = request.get_json(force=True)
    energy = data['Energy']
    liveness = data['Liveness']
    Positiveness = data['Positiveness']
    Loudness = data['Loudness']

    input_data = pd.DataFrame({'Energy': [energy], 'Liveness': [liveness], 'Positiveness': [Positiveness], 'Loudness': [Loudness]})
    
    # Memuat model dari file pickle
    model = load_model('knn_model.pkl')
    
    # Melakukan prediksi
    prediction = predict(model, input_data)
    
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO history (Energy, Liveness, Loudness, Positiveness,label) VALUES (%s, %s, %s, %s,%s)",
                (energy, liveness, Loudness, Positiveness,prediction[0]))
    mysql.connection.commit()
    cur.close()
    
    return jsonify({'prediction': prediction[0]})

@app.route('/get_history', methods=['GET'])
def get_history():
    try:
        cur = mysql.connection.cursor(cursorclass=DictCursor)  # Mengatur dictionary=True untuk mendapatkan hasil dalam bentuk dictionary
        cur.execute("SELECT * FROM history")
        data = cur.fetchall()
        cur.close()

        # Mengembalikan data dari database sebagai respons JSON
        history_list = []
        for row in data:
            history_dict = {
                'Energy': row['Energy'],
                'Liveness': row['Liveness'],
                'Positiveness': row['Positiveness'],
                'Loudness': row['Loudness'],
                'Label': row['label']  # Pastikan menggunakan 'Label' bukan 'label'
            }
            history_list.append(history_dict)

        return jsonify({'history': history_list})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/check_db_connection')
def check_db_connection():
    try:
        cur = mysql.connection.cursor()
        cur.execute('SELECT VERSION()')
        db_version = cur.fetchone()
        cur.close()
        
        return f"Connected to MySQL, version: {db_version[0]}"
    except Exception as e:
        return f"Failed to connect to MySQL: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)



import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import model_from_json
import joblib

app = Flask(__name__)

scaler_y = joblib.load('scaler.save')
json_file = open('model_test.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model_test.h5")
print("Loaded model from disk")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    input = request.form.values()
    lst = list(input)
    smile = lst[0]
    mol = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    array = np.zeros((0, ), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    final_features = array
    final_features = np.array([final_features])
    prediction = model.predict(final_features)
    prediction = scaler_y.inverse_transform(prediction)

    output = round(prediction[0][0], 2)

    return render_template('index.html', prediction_text='Predicted Ki Value: {}'.format(output)) # replace smile with output

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True, threaded=False)

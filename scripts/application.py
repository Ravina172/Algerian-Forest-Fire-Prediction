from flask import Flask, request, jsonify, render_template
import pickle
import pandas 
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_datapoint', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_scaled_data = standard_scaler.transform(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        )
        result = ridge_model.predict(new_scaled_data)
        return render_template('home.html', results=round(result[0], 2))
    # if it's a GET request, just show the form instead of error
    else:
     return render_template('index.html', results="No data found")


if __name__ == '__main__':
    app.run(debug=True)

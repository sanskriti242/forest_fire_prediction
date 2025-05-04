from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

## import ridge and standard scaler models


ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))

def get_risk_message(fwi):
    if fwi < 5:
        return "Low risk – Very little risk of a fire starting."
    elif 5 <= fwi < 12:
        return "Moderate risk – Fires can start, but will not spread quickly."
    elif 12 <= fwi < 20:
        return "High risk – Fires may spread rapidly, and suppression efforts may become difficult."
    elif 20 <= fwi < 40:
        return "Very high risk – Fires are expected to spread quickly, and extreme measures are required to contain them."
    else:
        return "Extreme risk – Conditions are ideal for rapidly spreading fires that can become uncontrollable."


@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = 0 if Temperature < 30 else 1
        Region = float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)
        fwi_value = float(result[0])
        message = get_risk_message(fwi_value)

        return render_template('home.html', result=fwi_value, message=message)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
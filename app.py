import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model from the pickle file
with open('modellll.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['Age'])
    gender = int(request.form['Gender'])
    sleep_duration = float(request.form['sleep_duration'])
    rem_percentage = int(request.form['rem_percentage'])
    deep_percentage = int(request.form['deep_percentage'])
    light_percentage = int(request.form['light_percentage'])
    awakenings = float(request.form['awakenings'])
    caffeine = float(request.form['caffeine'])
    alcohol = float(request.form['alcohol'])
    smoking = int(request.form['smoking'])
    exercise = float(request.form['exercise'])

    features = np.array([[age, gender, sleep_duration, rem_percentage, deep_percentage, light_percentage,
                          awakenings, caffeine, alcohol, smoking, exercise]])

    sleep_efficiency_prediction = model.predict(features)[0]
    
    return render_template('index.html', prediction=f'Predicted Sleep Efficiency: {sleep_efficiency_prediction:.2f}')

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import joblib
from feature_engineering import engineer_features

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        form_data = {
            'sl_no': request.form.get('sl_no'),
            'gender': request.form.get('gender'),
            'ssc_p': float(request.form.get('ssc_p')),
            'ssc_b': request.form.get('ssc_b'),
            'hsc_p': float(request.form.get('hsc_p')),
            'hsc_b': request.form.get('hsc_b'),
            'hsc_s': request.form.get('hsc_s'),
            'degree_p': float(request.form.get('degree_p')),
            'degree_t': request.form.get('degree_t'),
            'workex': request.form.get('workex'),
            'etest_p': float(request.form.get('etest_p')),
            'specialisation': request.form.get('specialisation'),
            'mba_p': float(request.form.get('mba_p')),
            'salary': float(request.form.get('salary'))
        }
        
        # Create DataFrame from form data
        df = pd.DataFrame([form_data])
        
        # Feature engineering
        df = engineer_features(df)
        
        # Predict
        prediction = model.predict(df)[0]
        label_mapping = {1: 'placed', 0: 'not placed'}
        prediction_label = label_mapping.get(prediction, 'Error')
        
        return redirect(url_for('result', prediction=prediction_label))
    except Exception as e:
        return redirect(url_for('error'))

@app.route('/result/<prediction>')
def result(prediction):
    return render_template('result.html', prediction=prediction)

@app.route('/error')
def error():
    return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)
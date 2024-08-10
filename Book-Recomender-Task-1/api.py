from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from model import build_model, get_recommendations
from preprocessing import preprocess_data
from database import initialize_client, load_dataset
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model and data
cosine_sim = joblib.load('model.joblib')
client = initialize_client("AstraCS:ZhWGRUYQWTligjvKudIArNia:40fa9a7b5964ae1bf2d7d1b0360c9825b383f4200f40d649c4d83e680e2a73c4",
                           "https://5df008b3-dfbe-4fbc-9703-a304374d9cc4-us-east-2.apps.astra.datastax.com")
df = load_dataset(client, "books")
df = preprocess_data(df)

# Pre-build the model
model = build_model(df)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form
    if 'Title' not in data:
        return jsonify({'error': 'Title is required'}), 400
    title = data['Title']
    # Ensure the function call matches the expected parameters
    recommendations = get_recommendations(title, df, cosine_sim)
    if request.is_json:
        return jsonify({'recommendations': recommendations})
    return render_template('index.html', title=title, recommendations=recommendations)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from joblib import load
from category_encoders import BinaryEncoder
import os

# Load model
model = load('decsiion_tree.joblib')  # or 'decision_tree.joblib' if you fix the name

# Load dataset
x = pd.read_csv('student_depression_dataset.csv')

# Prepare encoder
categorical_features = ['Gender', 'Profession', 'Dietary Habits', 
                        'Have you ever had suicidal thoughts ?', 
                        'Family History of Mental Illness', 'Sleep Duration', 'Financial Stress']
encoder = BinaryEncoder()
if all(feature in x.columns for feature in categorical_features):
    encoder.fit(x[categorical_features])

# Init Flask
api = Flask(__name__)
CORS(api)

# Serve HTML
@api.route('/')
def home():
    return render_template('depression_prediction.html')

# API endpoint
@api.route('/api/sd_prediction', methods=['POST'])
def prediction_depression():
    data = request.json['inputs']
    input_df = pd.DataFrame(data)

    input_encoded = encoder.transform(input_df[categorical_features])
    input_df = input_df.drop(categorical_features, axis=1)
    input_encoded = input_encoded.reset_index(drop=True)
    final_input = pd.concat([input_df, input_encoded], axis=1)

    prediction_proba = model.predict_proba(final_input)
    predictions = (prediction_proba[:, 1] > 0.5).astype(int)

    response = []
    for prob, pred in zip(prediction_proba, predictions):
        response.append({
            "Depression Probability": round(float(prob[1]) * 100, 2),
            "Prediction": "Likely to have depression" if pred == 1 else "Not likely to have depression"
        })

    return jsonify({"Prediction": response})

# Run app
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    api.run(debug=False, host='0.0.0.0', port=port)

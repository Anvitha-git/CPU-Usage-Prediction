import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_FILE = "model.pkl"
model = None

def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_FILE)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    load_model()
    try:
        data = request.get_json(force=True)
        
        # Ensure controller_kind exists for compatibility with older models
        if 'controller_kind' not in data:
            data['controller_kind'] = 'Deployment'

        df = pd.DataFrame([data])
        
        # One-hot encode controller_kind
        if 'controller_kind' in df.columns:
            df = pd.get_dummies(df, columns=['controller_kind'], drop_first=False)
        
        # Align columns with model
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder and select columns
        df = df[model_features]
        
        # Clip prediction to 0-1 range
        raw_pred = model.predict(df)[0]
        prediction = max(0.0, min(1.0, raw_pred))
        
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5003)

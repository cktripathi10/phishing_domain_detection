from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html')  # Ensure this template has the correct form as described previously

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  # This should contain the form for phishing domain data input
    else:
        # Assuming we need to collect various inputs for phishing prediction
        url = request.form.get('url')
        ssl_state = request.form.get('ssl_state')
        domain_registration_length = int(request.form.get('domain_registration_length'))
        https_token = request.form.get('https_token')
        request_url = int(request.form.get('request_url'))

        # Create a DataFrame or a suitable data structure for prediction
        input_data = pd.DataFrame({
            'url': [url],
            'ssl_state': [ssl_state],
            'domain_registration_length': [domain_registration_length],
            'https_token': [https_token],
            'request_url': [request_url]
        })

        # Here, you need to preprocess input_data as per your model's needs
        print(input_data)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(input_data)
        print("After Prediction")

        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)  # Added debug=True for development purposes

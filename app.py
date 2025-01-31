from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_data', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else: 
        data = CustomData(
            odor = request.form.get("odor"),
            gill_color = request.form.get("gill_color"),
            spore_print_color = request.form.get("spore_print_color"),
            gill_size = request.form.get("gill_size"),
            bruises = request.form.get("bruises")
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)
        if pred == 0:
            results = "Edible"
        elif pred == 1:
            results = "Poisonous"
        else:
            results = "Not match"
        return render_template('home.html', results=results)
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline. predict_pipeline import CustomData, PredictPipeline

from sklearn.preprocessing import OrdinalEncoder

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
            cap_shape= request.form.get("cap_shape"),
            cap_surface = request.form.get("cap_surface"),
            cap_color = request.form.get("cap_color"),
            bruises = request.form.get("bruises"),
            odor = request.form.get("odor"),
            gill_attachment = request.form.get("gill_attachment"),
            gill_spacing = request.form.get("gill_spacing"),
            gill_size = request.form.get("gill_size"),
            gill_color = request.form.get("gill_color"),
            stalk_shape = request.form.get("stalk_shape"),
            stalk_root = request.form.get("stalk_root"),
            stalk_surface_above_ring = request.form.get("stalk_surface_above_ring"),
            stalk_surface_below_ring = request.form.get("stalk_surface_below_ring"),
            stalk_color_above_ring = request.form.get("stalk_color_above_ring"),
            stalk_color_below_ring = request.form.get("stalk_color_below_ring"),
            veil_type = request.form.get("veil_type"),
            veil_color = request.form.get("veil_color"),
            ring_number = request.form.get("ring_number"),
            ring_type = request.form.get("ring_type"),
            spore_print_color = request.form.get("spore_print_color"),
            population = request.form.get("population"),
            habitat = request.form.get("habitat")
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000)
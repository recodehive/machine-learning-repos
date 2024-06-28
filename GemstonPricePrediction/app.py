from flask import Flask, request, render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def home_page():
    return render_template('test.html')
@app.route('/predict',methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('test.html')
    
    else:
        data = CustomData(
            carat = float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get('cut'),
            color = request.form.get('color'),
            clarity= request.form.get('clarity')
        )

        final_new_data = data.get_data_as_datarframe()
        predict_Pipeline = PredictPipeline()
        pred = predict_Pipeline.predict(final_new_data)

        results = "₹"+" " +(round(pred[0],2).astype(str))
        return render_template('test.html',final_result = results)



if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
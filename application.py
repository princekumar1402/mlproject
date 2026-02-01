from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app=application
# Landing page
@app.route("/")
def landing():
    return render_template("landing.html")

# Prediction form page
@app.route("/predict")
def predict_page():
    return render_template("predict.html")

# Prediction result
@app.route("/predictdata", methods=["POST"])
def predict():
    data = CustomData(
        gender=request.form["gender"],
        race_ethnicity=request.form["ethnicity"],
        parental_level_of_education=request.form["parental_level_of_education"],
        lunch=request.form["lunch"],
        test_preparation_course=request.form["test_preparation_course"],
        reading_score=float(request.form["reading_score"]),
        writing_score=float(request.form["writing_score"])
    )

    pred_df = data.get_data_as_data_frame()
    preds = PredictPipeline().predict(pred_df)

    return render_template("home.html", results=round(preds[0], 2))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


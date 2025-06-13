from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# loading the ML trained model
with open('model.pkl', 'rb') as obj:
    model = pickle.load(obj)


def get_cleaned_data(form_data):
    gestation = float(form_data['gestation'])
    parity = int(form_data['parity'])
    age = float(form_data['age'])
    height = float(form_data['height'])
    weight = float(form_data['weight'])
    smoke = float(form_data['smoke'])

    cleaned_data = {
        "gestation": [gestation],
        "parity": [parity],
        "age": [age],
        "height": [height],
        "weight": [weight],
        "smoke": [smoke]
    }
    return cleaned_data


@app.route('/')
def form_component():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def get_prediction():
    # get the data from user
    baby_data_form = request.form

    baby_data_cleaned = get_cleaned_data(baby_data_form)

    # converting baby json data into a data frame
    baby_df = pd.DataFrame(baby_data_cleaned)

    predicted_wgt = model.predict(baby_df)
    predicted_wgt = round(float(predicted_wgt), 2)

    # returning a response in json format
    response = {"Prediction": predicted_wgt}
    return render_template("index.html", predicted_wgt=predicted_wgt)


if __name__ == "__main__":
    app.run(debug=True)

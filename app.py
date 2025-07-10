from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and label encoder
model = pickle.load(open('model.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

@app.route('/')
def home():
    neighborhoods = le.classes_  # List of neighborhood names
    return render_template('index.html', neighborhoods=neighborhoods)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        overall_qual = int(request.form['OverallQual'])
        gr_liv_area = int(request.form['GrLivArea'])
        year_built = int(request.form['YearBuilt'])
        garage_cars = int(request.form['GarageCars'])
        neighborhood = request.form['Neighborhood']

        encoded_neigh = le.transform([neighborhood])[0]

        features = np.array([[overall_qual, gr_liv_area, year_built, garage_cars, encoded_neigh]])
        prediction = model.predict(features)[0]

        return render_template('index.html',
                               prediction_text=f"🏡 Estimated House Price: ₹{prediction:,.2f}",
                               neighborhoods=le.classes_)
    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"❌ Error: {str(e)}",
                               neighborhoods=le.classes_)

if __name__ == "__main__":
    app.run(debug=True)

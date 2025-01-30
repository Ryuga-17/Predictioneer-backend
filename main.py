from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
import joblib
app = Flask(__name__)
CORS(app)
# with open('crf2.pkl', 'rb') as f:
#     cfr_model = pickle.load(f)

# with open('deaths2.pkl', 'rb') as f:
#     deaths_model = pickle.load(f)

cfr_model=joblib.load('crf2.pkl')
deaths_model=joblib.load('deaths2.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    lat = data['lat']
    lng = data['lng']
    cfr = cfr_model.predict([[lat, lng]])[0]
    deaths = deaths_model.predict([[lat, lng]])[0]
    
    return jsonify({
        'cfr': float(cfr),
        'deaths': int(deaths)
    })

if __name__ == '__main__':
    app.run(debug=True)


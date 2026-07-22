import mlflow
from flask import Flask, request, jsonify

MLFLOW_TRACKING_URI = 'http://localhost:5000'
RUN_ID = '38ee7005f2d64eff8f1e2edad80cd244'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Not hosted in S3 so we need to manually load the model from the run
logged_model = f"runs:/{RUN_ID}/model"
model = mlflow.pyfunc.load_model(logged_model)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = f"{ride['PUlocationID']}_{ride['DOlocationID']}"
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    preds = model.predict(features)
    return preds[0]

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'prediction': float(pred),
        'run_id': RUN_ID,
        'model_type': 'random_forest',
        'model_version': 'v1'
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
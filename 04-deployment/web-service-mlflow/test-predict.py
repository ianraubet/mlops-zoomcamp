import predict

ride = {
    "PUlocationID": 10,
    "DOlocationID": 50,
    "trip_distance": 40
}

features = predict.prepare_features(ride)
pred = predict.predict(features)

print(pred)
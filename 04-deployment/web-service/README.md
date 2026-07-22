## Deploying a model as a web-service

* Using uv to manage the venv
* Creating a script for predicting
* Placing the script into a Flask app
* Packaging the app with Docker

```bash
docker build -t ride-duration-prediction-service:v1 .
```

```bash
docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1
```
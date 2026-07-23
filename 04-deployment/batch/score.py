import argparse

import mlflow
import os
import pandas as pd
import sys
import uuid

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline

def month_type(value):
    ivalue = int(value)
    if not 1 <= ivalue <= 12:
        raise argparse.ArgumentTypeError(f"month must be between 1 and 12, got {ivalue}")
    return ivalue


def year_type(value):
    ivalue = int(value)
    if not 2009 <= ivalue <= 2026:
        raise argparse.ArgumentTypeError(f"year must be between 2009 and 2026, got {ivalue}")
    return ivalue


def generate_uuids(n):
    ride_ids = []

    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))

    return ride_ids


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    df['ride_id'] = generate_uuids(len(df))

    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts


def load_model(run_id):
    # Not hosted in S3 so we need to manually load the model from the run
    logged_model = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(logged_model)

    return model


def compute_results(df, y_pred, run_id):
    df_result = pd.DataFrame()

    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id

    return df_result


def apply_model(input_file, run_id, output_file):
    print(f"Reading input file: {input_file}")
    df = read_dataframe(input_file)

    print(f"Preparing dictionaries for {len(df)} records.")
    dicts = prepare_dictionaries(df)

    print(f"Applying model from run_id: {run_id} to {len(dicts)} records.")
    model = load_model(run_id)

    print(f"Computing predictions for {len(dicts)} records.")
    y_pred = model.predict(dicts)

    print(f"Predictions computed for {len(y_pred)} records.")
    computed_results = compute_results(df, y_pred, run_id)

    print(f"Saving results to: {output_file}")
    computed_results.to_parquet(output_file, index=False)


def run():
    argparser = argparse.ArgumentParser(description='Apply model to input data and save results.')
    argparser.add_argument(
        '--taxi_type',
        type=str,
        choices=['green', 'yellow'],
        required=True,
        help='Type of taxi (green or yellow)'
    )
    argparser.add_argument(
        '--year',
        type=year_type,
        required=True,
        help='Year of the data'
    )
    argparser.add_argument(
        '--month',
        type=month_type,
        required=True,
        help='Month of the data'
    )
    argparser.add_argument('--mlflow_tracking_uri', type=str, default='http://localhost:5000', help='MLflow tracking URI')
    argparser.add_argument('--run_id', type=str, default='38ee7005f2d64eff8f1e2edad80cd244', help='MLflow run ID of the model to use')
    args = argparser.parse_args()
    
    taxi_type = args.taxi_type
    year = args.year
    month = args.month
    mlflow_tracking_uri = args.mlflow_tracking_uri
    run_id = args.run_id

    os.makedirs('output', exist_ok=True)
    os.makedirs(f'output/{taxi_type}', exist_ok=True)

    print(f"Applying model from run_id: {run_id} to {taxi_type} taxi data for {year}-{month:02d}")
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    apply_model(
        input_file=input_file, 
        run_id=run_id, 
        output_file=output_file
    )


if __name__ == "__main__":
    run()
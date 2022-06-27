#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import sys
import uuid

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids

def run():
    year = sys.argv[1]
    month = sys.argv[2]
    print(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year}-0{month}.parquet')
    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year}-0{month}.parquet')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    #Print the y_pred mean
    print('Y_pred value is: ',y_pred.mean())
    df['ride_id'] = generate_uuids(len(df))
    # write the ride id and the predictions to a dataframe with results
    df_result = pd.DataFrame({'ride_id': df.ride_id, 'prediction': y_pred})

    # Saving the results of the model into a parquet file.
    df_result.to_parquet(
        'output_file_03_21',
        engine='pyarrow',
        compression=None,
        index=False
    )

if __name__ == '__main__':
    run()
'''
(base) ubuntu@ip-172-31-4-164:~/mlops-zoomcamp/04-deployment/homework$ python starter.py 2022 3
https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_2022-03.parquet
Y_pred value is:  17.085477922673828
(base) ubuntu@ip-172-31-4-164:~/mlops-zoomcamp/04-deployment/homework$

'''
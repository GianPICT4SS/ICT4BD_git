"""This is a fake publisher: it task is to read data from eplusout.csv file and publishing them via MQTT, another fake
subscriber reads this data and save it on influxdb. The data to be publish are 24*365 sample, the data of one year, one
per hour. It will be publish 24 samples each minute so that the simulation will have a duration of six hours."""

import time
import json
import datetime
import pandas as pd


from src.method_building import Prediction
from src.predictive_model.mqtt_building import Building_publisher

learn = Prediction()


df = pd.read_csv('../eplus_simulation/eplus/eplusout.csv')
print('DATASET CARICATO')
# Create a user csv
df = learn.create_csv(df)
columns = df.columns
##################################################à
# MQTT Publisher
###################################################
pub = Building_publisher(clientID='Office_ON', topic='OFFICE', qos=0)
pub.start()
dict_ = {'Payload': {}}


for i in range(df.shape[0]-1):
    row = df.iloc[i:i+1, :]
    date = row.index.format('str')[1]
    dict_['Payload']['Time'] = date
    for col in columns:
        dict_['Payload'][col] = row[col].to_json()
    msg = json.dumps(dict_)
    pub.publish(msg=msg)
    time.sleep(1.5)
pub.stop()



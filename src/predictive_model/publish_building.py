"""This is a fake publisher: its task is to read data from eplusout.csv file and publishing them via MQTT, another fake
subscriber reads this data and save it on influxdb. The data to be published are 24*6*365 samples, the data of one year, one
per 10 minutes.
"""

import time
import json
import datetime
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(1,'../')

from method_building import Prediction
from predictive_model.mqtt_building import Building_publisher

learn = Prediction()

csv_path = Path('../eplus_simulation/eplus/eplusout_Office_On_corrected_Optimal.csv')
df = pd.read_csv(csv_path)
print(df.shape[0])
# Create a user csv
df = learn.create_csv(df)
columns = df.columns
##################################################à
# MQTT Publisher
###################################################
pub = Building_publisher(clientID='pub_Office_ON', topic='OFFICE', qos=2)
pub.start()
pub.mqtt_client.loop_start()
dict_ = {'Payload': {}}
for i in range(df.shape[0]-1):
	row = df.iloc[i, :]
	date = str(row['Date'])
	dict_['Payload']['Time'] = date
	for col in columns:
		if col != 'Date':
			dict_['Payload'][col] = row[col]
	msg = json.dumps(dict_, indent=4)
	pub.publish(msg=msg)
	# pub.mqtt_client.loop()
	time.sleep(0.5)
pub.stop()


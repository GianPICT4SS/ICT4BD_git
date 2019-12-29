"""This is a fake publisher: it task is to read data from eplusout.csv file and publishing them via MQTT, another fake
subscriber reads this data and save it on influxdb. The data to be publish are 24*365 sample, the data of one year, one
per hour. It will be publish 24 samples each minute so that the simulation will have a duration of six hours."""

import time
import json
import pandas as pd
import numpy as np
from retrofit import create_csv
from building_mqtt import Building_publisher


df = pd.read_csv('eplus/eplusout.csv')
df_t = create_csv(df)
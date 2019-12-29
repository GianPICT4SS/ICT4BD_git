import paho.mqtt.client as mqtt
import logging
from datetime import datetime
import time
import json
from mqtt_building import Building_subscriber
from influxdb import InfluxDBClient





client = InfluxDBClient(host='localhost', port=8086)

client.create_database('pyexample')

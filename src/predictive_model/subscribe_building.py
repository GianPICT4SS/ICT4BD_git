import paho.mqtt.client as mqtt
import logging
from datetime import datetime
import time
import json
from mqtt_building import Building_subscriber
from influxdb import InfluxDBClient

class InfluxDB:
	def __init__(self, database_name='BuildingDesign'):
		self.client = InfluxDBClient(host='localhost', port=8086)
		if(not database_name in self.get_databases()):
			self.client.create_database(database_name)

		self.client.switch_database(database_name)

	def get_databases(self):
		dbs = self.client.get_list_database()
		l = [i['name'] for i in dbs]
		return l

	def format_json_body(self, payload):
		json_body = []
		if 'Payload' in payload.keys():
			body = payload['Payload']
			time = body['Time']
			list_measurements = list(body.keys())
			list_measurements.remove('Time')
			for k in list_measurements:
				tmp_object = {}
				tmp_object['measurement'] = str(k)
				tmp_body = json.loads(body[k])
				timestamp = int(list(tmp_body.keys())[-1])//1000
				new_time =  datetime.fromtimestamp(timestamp)
				new_value = list(tmp_body.values())[-1]
				tmp_object['time'] = new_time
				tmp_object['fields'] = {
					'value': new_value
				}
				json_body.append(tmp_object)
		return json_body

	def run(self):
		sub = Building_subscriber(clientID='sub_Office_ON', topic='OFFICE', qos=0)
		sub.start()
		i = 1
		body_list = []
		while i:
			sub.subscribe()
			body = sub.msg_body
			json_body = self.format_json_body(body)
			self.client.write_points(json_body)

if __name__ == '__main__':
	influx = InfluxDB()
	influx.run()
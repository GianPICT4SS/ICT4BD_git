import paho.mqtt.client as mqtt
import logging
from datetime import datetime
import time
import json
from mqtt_building import Building_subscriber
from influxdb import InfluxDBClient, DataFrameClient
import pandas as pd

class InfluxDB:
	def __init__(self, database_name='BuildingDesign', drop=0):
		self.db_name = database_name
		self.client = InfluxDBClient(host='localhost', port=8086)
		# if drop:
		# 	self.client.drop_database(self.db_name)
		if(not database_name in self.get_databases()):
			self.client.create_database(database_name)

		self.client.switch_database(database_name)

	def get_databases(self):
		dbs = self.client.get_list_database()
		l = [i['name'] for i in dbs]
		return l

	def format_json_body(self, payload, past):
		json_body = []
		if 'Payload' in payload.keys():
			body = payload['Payload']
			time = body['Time']
			t = datetime.strptime(body['Time'], '%Y-%m-%d %H:%M:%S')
			if t > past:
				print(t-past)
				past = t
				list_measurements = list(body.keys())
				list_measurements.remove('Time')
				for k in list_measurements:
					tmp_object = {}
					tmp_object['measurement'] = str(k)
					tmp_object['time'] = time
					tmp_object['fields'] = {
						'value': body[k]
					}
					json_body.append(tmp_object)

		return json_body, past

	def run(self):
		sub = Building_subscriber(clientID='sub_Office_ON', topic='OFFICE', qos=2)
		sub.start()
		i = 1
		body_list = []
		time = datetime(2010,1,1)
		# while i:
		# 	sub.subscribe()
		# 	body = sub.msg_body
		# 	json_body = self.format_json_body(body)
		# 	if len(json_body) > 0: 
		# 		self.client.write_points(json_body)
			# time.sleep(2)

		sub.subscribe()
		while i:
			body = sub.msg_body
			json_body, time = self.format_json_body(body, time)
			if(len(json_body)>0):
				self.client.write_points(json_body)
		# sub.mqtt_client.loop_forever()


	def get_dataframe(self):
		df = {}
		measurements = self.client.get_list_measurements()
		for m in measurements:
			# print(m)
			name = m['name']
			q = f'select "value" from "{name}"'
			query = self.client.query(q)
			data = [i[1] for i in query.raw['series'][0]['values']]
			index = [i[0] for i in query.raw['series'][0]['values']]
			if 'Date' not in df.keys():
				df['Date'] = index
			df[name] = data
		df = pd.DataFrame(df)
		df.set_index(df['Date'])
		
		return df


if __name__ == '__main__':
	influx = InfluxDB()
	# influx.run()
	influx.get_dataframe()
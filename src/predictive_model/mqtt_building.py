import paho.mqtt.client as mqtt
import logging
from datetime import datetime
import time
import json
from queue import Queue


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                    datefmt='%d/%m/%Y %H:%M ',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

class Building_publisher():
    def __init__(self, clientID, broker="mqtt.eclipse.org", port=1883, topic="", qos=1):
        self.clientID_ = clientID
        self.broker_ = broker
        self.port_ = port
        self.mqtt_client = mqtt.Client(self.clientID_, clean_session=False)
        self.topic_ = topic
        self.qos_ = qos
        self.queue_ = Queue()

        self.mqtt_client.on_connect = self.OnConnect
        self.mqtt_client.on_disconnect = self.OnDisconnect
        logger.info(f"User {self.clientID_} initialized.")

    def OnConnect(self, mqtt_client, userdata, flags, rc):
        """ myOnConnect function called by on_connect callback:

        Called upon connection to the broker. Everything goes well if rc == 0
        otherwise we have some connection issues with the broker. If so it is
        printed in the terminal and the notify() method of the notifier is
        called so that an appropriate action can be taken.

        Args:
            mqtt_client (:obj: MQTT.Client): client instance of the callback
            userdata (str): user data as set in Client (not used here)
            flags (int): flag to notify if the user's session is still
                available (not used here)
            rc (int): result code
        """
        errMsg = ""

        if rc == 0:
            logger.info(f"MQTT client {self.clientID_} successfully connected to broker!")
            return str(rc)

        # If we go through this we had a problem with the connection phase
        elif 0 < rc <= 5:
            errMsg = "/!\ " + self.clientID_ + " connection to broker was " \
                                              "refused because of: "
            if rc == 1:
                errMsg.append("the use of an incorrect protocol version!")
            elif rc == 2:
                errMsg.append("the use of an invalid client identifier!")
            elif rc == 3:
                errMsg.append("the server is unavailable!")
            elif rc == 4:
                errMsg.append("the use of a bad username or password!")
            else:
                errMsg.append("it was not authorised!")
        else:
            errMsg = "/!\ " + self.clientID_ + " connection to broker was " \
                                              "refused for unknown reasons!"
        logger.error(errMsg)


    def publish(self, msg):
        """ myPublish:

                Method that makes the MQTT client publish to the the broker a message
                under a specific topic and with a particular QoS, which by default is 2.

                Args:
                    topic (str): topic to which you desire to publish
                    msg (str): message you wish to publish
                    qos (int, optional): desired QoS, default to 2
                """
        logger.info(f"MQTT client {self.clientID_} publishing {msg} with topic {self.topic_}.")
        # publish a message with a certain topic

        self.mqtt_client.publish(self.topic_, msg, self.qos_)
        # self.mqtt_client.loop()

    def OnDisconnect(self, mqtt_client, userdata, rc):
        """ myOnDisconnect function called by on_disconnect callback:

        Can be triggered in one of two cases:
        - in response to a disconnect(): normal case, it was asked
        - in response to an unexpected disconnection: in that case the client
        will try to reconnect

        In both cases we log it.

        Args:
            mqtt_client (:obj: MQTT.Client): client instance of the callback
            userdata (str): user data as set in Client (not used here)
            rc (int): result code
        """
        if rc == 0:
            logger.info(f"MQTT client {self.clientID_} successfully disconnected!")
        else:
            logger.warning(f"Unexpected disconnection of MQTT client {self.clientID_}. "\
                           "Reconnecting right away!")
            # The reconnection is performed automatically by our client since
            # we're using loop_start() so no need to manually tell our client
            # to reconnect.

    def start(self):
        self.mqtt_client.connect(host=self.broker_, port=self.port_)
        self.mqtt_client.loop_start()
        logger.info(f"Client {self.clientID_} connected to the Broker {self.broker_}")

    def stop(self):
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        logger.info(f"Client {self.clientID_} disconnected from the Broker")


class Building_subscriber():
    """Class Led_subscriber: its instances can connect to the broker and subscriber to some topic"""

    # msg_body = {}
    def __init__(self, clientID, broker="mqtt.eclipse.org", port=1883, topic="", qos=1):
        self.msg_body = {}
        self.clientID_ = clientID
        self.broker_ = broker
        self.port_ = port
        self.mqtt_client = mqtt.Client(self.clientID_, clean_session=False)
        self.isSubscribe_ = False
        self.topic_ = topic
        self.qos_ = qos
        #self.queue_ = Queue()

        self.mqtt_client.on_connect = self.OnConnect
        self.mqtt_client.on_message = self.OnMessage
        self.mqtt_client.on_disconnect = self.OnDisconnect
        logger.info(f"User {self.clientID_} initialized.")

    def OnConnect(self, mqtt_client, userdata, flags, rc):
        """ myOnConnect function called by on_connect callback:

        Called upon connection to the broker. Everything goes well if rc == 0
        otherwise we have some connection issues with the broker. If so it is
        printed in the terminal and the notify() method of the notifier is
        called so that an appropriate action can be taken.

        Args:
            mqtt_client (:obj: MQTT.Client): client instance of the callback
            userdata (str): user data as set in Client (not used here)
            flags (int): flag to notify if the user's session is still
                available (not used here)
            rc (int): result code
        """
        errMsg = ""

        if rc == 0:
            logger.info(f"MQTT client {self.clientID_} successfully connected to the Broker {self.broker_}!")
            return str(rc)

        # If we go through this we had a problem with the connection phase
        elif 0 < rc <= 5:
            errMsg = "/!\ " + self.clientID_ + " connection to broker was " \
                                              "refused because of: "
            if rc == 1:
                errMsg.append("the use of an incorrect protocol version!")
            elif rc == 2:
                errMsg.append("the use of an invalid client identifier!")
            elif rc == 3:
                errMsg.append("the server is unavailable!")
            elif rc == 4:
                errMsg.append("the use of a bad username or password!")
            else:
                errMsg.append("it was not authorised!")
        else:
            errMsg = "/!\ " + self.clientID_ + " connection to broker was " \
                                              "refused for unknown reasons!"
        logger.error(errMsg)


    def subscribe(self):
        self.mqtt_client.subscribe(topic=self.topic_, qos=self.qos_)
        self.isSubscribe_ = True
        logger.info(f"Client {self.clientID_} subscribe to the topic: {self.topic_}")

    def OnMessage(self, client, userdata, msg):
        # global msg_body
        get_time = datetime.now()
        current_time = get_time.strftime("%Y-%m-%d %H:%M:%S")
        # print("--------------------------------------------------------------------")
        # print("message received: ", str(msg.payload.decode("utf-8")))
        # print("at time: " + str(current_time))
        # print("--------------------------------------------------------------------")
        self.msg_body = json.loads(msg.payload)

    def OnDisconnect(self, mqtt_client, userdata, rc):
        """ myOnDisconnect function called by on_disconnect callback:

        Can be triggered in one of two cases:
        - in response to a disconnect(): normal case, it was asked
        - in response to an unexpected disconnection: in that case the client
        will try to reconnect

        In both cases we log it.

        Args:
            mqtt_client (:obj: MQTT.Client): client instance of the callback
            userdata (str): user data as set in Client (not used here)
            rc (int): result code
        """
        if rc == 0:
            logger.info(f"MQTT client {self.clientID_} successfully disconnected!")
        else:
            logger.warning(f"Unexpected disconnection of MQTT client {self.clientID_}. "\
                           "Reconnecting right away!")
            # The reconnection is performed automatically by our client since
            # we're using loop_start() so no need to manually tell our client
            # to reconnect.

    def start(self):

        self.mqtt_client.connect(host=self.broker_, port=self.port_)
        self.mqtt_client.loop_start()
        #logger.info(f"Client {self.clientID_} connected to the Broker {self.broker_}")

    def stop(self):

        if self.isSubscribe_:
            self.mqtt_client.unsubscribe(self.topic_)
            logger.info(f"MQTT client {self.clientID_} unsubscribed from topic {self.topic_}.")

        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        logger.info(f"Client {self.clientID_} disconnected from the Broker")


if __name__ == "__main__":

    pub = Building_publisher(clientID="pub_1", topic="Building", qos=2)
    start = int(datetime.now().strftime("%H:%M"))
    i = True
    pub.start()
    while i:
        end = int(datetime.now().strftime("%H:%M"))
        end_ = datetime.now().strftime("%Y-%m-%d" + "|" + "%H:%M:%S")
        dict_OFF = {'Time': end_, 'Payload': 0}
        dict_ON = {'Time': end_, 'Payload': 1}
        if end-start >= 1:
            msg_ON = json.dumps(dict_ON)
            pub.publish(msg=msg_ON)
            time.sleep(.5)
            i = False
        else:
            msg_OFF = json.dumps(dict_OFF)
            pub.publish(msg=msg_OFF)
            time.sleep(.5)
    pub.stop()
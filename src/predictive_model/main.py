from RecurrentNeuralNetwork import *

rnn = RecurrentNeuralNetwork()
df, features = rnn.return_features()
#6 sample for each hour --> 6*24 = 1 day
epochs = 50
mode = '3-output'
step = 1
past = 1
future = 12

name = f"Model_epochs{epochs}_history{past}_future{future}_step{step}_mode{mode}"
model_name = f"{name}.h5"
json_name = f"{name}.json"

models = os.listdir('models/')
if json_name not in models:	
	rnn.create_model(df, features, look_back=past, future=future, epochs=epochs, mode=mode, step=step, name=name)
else:
	model = rnn.load_model(json_name, model_name)
	rnn.create_model(df, features, look_back=past, future=future, epochs=epochs, train=0, load_model=model, mode=mode, step=step, name=name)
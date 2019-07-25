import numpy as np
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
correct_output = np.array([[0],[1],[1],[0]])

inputlayer = 2
hiddenlayer = 2
outputlayer  = 1

hiddenw = np.random.uniform(size=(inputlayer,hiddenlayer))
hidden_bias = np.random.uniform(size=(1,hiddenlayer))

outputw = np.random.uniform(size=(hiddenlayer,outputlayer))
output_bias = np.random.uniform(size=(1,outputlayer))

def sigmoid(x):
  return 1/(1+np.exp(-x))

act = np.dot(inputs,hiddenw)
act += hidden_bias

hiddenlayerout = sigmoid(act)



actout = np.dot(hiddenlayerout,outputw)
actout += output_bias
output = sigmoid(actout)

#neural network initalization completed 
#backprop starts here

epochs = 100000

lr = 0.1

def sigmoid_derivative(x):
    return x * (1 - x)

for _ in range(epochs):
	#Forward Propagation
	act = np.dot(inputs,hiddenw)
	act += hidden_bias
	hiddenlayerout = sigmoid(act)

	actout = np.dot(hiddenlayerout,outputw)
	actout += output_bias
	output = sigmoid(actout)

	#Backpropagation
	error = correct_output - output
	d_predicted_output = error * sigmoid_derivative(output)
	
	error_hidden_layer = d_predicted_output.dot(outputw.T)
	d_hidden_layer = error_hidden_layer * sigmoid_derivative(hiddenlayerout)

	#Updating Weights and Biases
	outputw += hiddenlayerout.T.dot(d_predicted_output) * lr
	output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
	hiddenw += inputs.T.dot(d_hidden_layer) * lr
	hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr

#training part done now lets take the inputs
 
x, y, z = input("Enter space seperated numbers for the gate  ").split() 
test = np.array([x,y])
import math
test = test.astype(int)
act = np.dot(test,hiddenw)
act = act + hidden_bias
hiddenlayerout = sigmoid(act)
actout = np.dot(hiddenlayerout,outputw)
actout += output_bias
output = sigmoid(actout)
#print(round(float(output)))
if z== '0' :
    print(round(float(output)))
else :
    print(round(abs(1-float(output))))
#print(round(abs(1-float(output))))        

#print(round(float(output)))
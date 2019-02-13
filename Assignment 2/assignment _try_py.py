"""
Aashish Adhikari
"""


from __future__ import division
from __future__ import print_function

import sys
import time
try:
   import _pickle as pickle
except:
   import pickle

import numpy as np
import math


class LinearTransform(object):

    def __init__(self, W, b): #linear transform ma kina weight diyo budo le bhaneko ta duit hau ma obj banauda farak farak
                                #diyera banauna ko lagi rahexa
        self.weights = W #weights for all n nodes in the hidden layer
        self.bias = b



    def forward(self, x):
        #print("aa",x.shape,self.weights.shape)
        #Never do this because it does element-wise and this gives an error
        #batch_linear_summation = x * self.weights# + self.bias


        batch_linear_summation = np.dot(x, self.weights) + self.bias
        #print("batch_linear_summation is ",batch_linear_summation)

        return (batch_linear_summation)#,self.relu_object.forward(batch_linear_summation_without_relu,y))

    def forward_2(self,x):
        batch_linear_summation_without_sigmoid = np.dot(x,self.weights)+self.bias
        return (batch_linear_summation_without_sigmoid,self.sig.forward(batch_linear_summation_without_sigmoid))
        print()
    def backward(self, grad_output,learning_rate,momentum,l2_penalty,weights_to_update,bias_to_update):

        dE_by_d_ip_of_this = grad_output * weights_to_update

        return dE_by_d_ip_of_this





        print()
	# DEFINE backward function


class ReLU(object):
    def __init__(self):
        print()

    def forward(self, x):
	# DEFINE forward function
        relu_output = np.maximum(0,x)
        #print("relu_output is",relu_output)
        return relu_output


    def backward(
        self,
        grad_output,
        learning_rate=0.0,
        momentum=0.0,
        l2_penalty=0.0,
    ):
        print()
    # DEFINE backward function
# ADD other operations in ReLU if needed



# This is a class for a sigmoid layer followed by a cross entropy layer, the reason
# this is put into a single layer is because it has a simple gradient form



class SigmoidCrossEntropy(object):
    def __init__(self):
        print()
        self.input_values=[]
        #self.linear_transform_object_2 = LinearTransform(self.second_layer_weights,self.second_layer_bias)

    def forward(self, x):
#         linear_summation = self.linear_transform_object_2.forward(self,x)
#         print("linear summation size ",linear_summation.shape)
#         #linear transform gar paile
        #ani balla sigmoid



        #print("sigmoid ma aayeko input ", x)
        self.input_values = x
        sigmoid =  (1 / (1 + np.exp(-x)))
        #print("sigmoid is ", sigmoid)
        return sigmoid

    def backward(self,input, true_output, predicted_output, grad_output, learning_rate,direction, momentum, l2_penalty,second_layer_wts, second_layer_bias ):


        grad_err_wrt_sigmoid_ip = true_output-predicted_output

        direction_updated_weights_2 = momentum * direction - np.dot(learning_rate, ( grad_err_wrt_sigmoid_ip + np.dot(l2_penalty, second_layer_wts )))

        direction_updated_bias_2 = momentum * direction - np.dot(learning_rate, ( grad_err_wrt_sigmoid_ip + np.dot(l2_penalty, second_layer_bias )))

        print("direction updated orr weight update is ",direction_updated_weights_2)
        print("bias direction updated orr bias update is ",direction_updated_bias_2)

        #-----------------------------------------------
        #call the previous layer for updates on the previous layer weights

        backward_return_for_sencond_linear_transform = self.linear_transform_object_for_sigmoid.backward(self,grad_output, learning_rate,direction, momentum, l2_penalty,second_layer_wts,second_layer_bias)


        return(direction_updated_weights_2, direction_updated_bias_2, backward_return_for_sencond_linear_transform )










# This is a class for the Multilayer perceptron




class MLP(object):

    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network

        self.input_dims = input_dims
        self.hidden_units = hidden_units


        self.linear_transform_layer = LinearTransform(np.random.rand(input_dims, hidden_units),np.random.rand(hidden_units))

        self.relu_layer = ReLU()

        self.linear_transform_object_for_sigmoid = LinearTransform(np.random.rand(hidden_units,1),np.random.rand(1))



        self.sigmoid_object_layer = SigmoidCrossEntropy()




        #------------------------------------
        self.x_for_this_batch = []
        self.y_for_this_batch = []
        self.y_cap_for_this_batch = []


        self.hidden_layer_output = [] #Relu through aayeko xa sigmoid of A1
        self.sigmoid_output = []#Sigmoid through aayeko is Relu of A2, equals y cap for this batch

        self.hidden_layer_op_before_Relu = [] #A1
        self.sigmoid_layer_op_before_sigmoid = []#A2
        #------------------


    def train(self,x_batch,y_batch,learning_rate,momentum,l2_penalty):

        self.x_for_this_batch = np.array(x_batch)
        self.y_for_this_batch = np.array(y_batch)
        #print("within train x_batch, y_batch",x_batch,y_batch)
        #print("x batch and y batch ",len(x_batch),len(y_batch))


        ###########------------First FOrward Pass and then backward pass---------




        self.hidden_layer_op_before_Relu= np.array(self.linear_transform_layer.forward(x_batch))

        self.hidden_layer_output = np.array(self.relu_layer.forward(self.hidden_layer_op_before_Relu))


        self.sigmoid_layer_linear_transform_op = np.array(self.linear_transform_object_for_sigmoid.forward(self.hidden_layer_output))

        self.sigmoid_layer_op = np.array(self.sigmoid_object_layer.forward(self.sigmoid_layer_linear_transform_op))


        #avoid overflow and underflow

        self.sigmoid_layer_op = np.where(self.sigmoid_layer_op == 0,0.00000001,self.sigmoid_layer_op)


        self.sigmoid_layer_op = np.where(self.sigmoid_layer_op == 1,1-0.00000001,self.sigmoid_layer_op)

        #self.y_cap_for_this_batch = self.sigmoid_layer_op


#         print("y_cap or sigmoid_output",self.sigmoid_layer_op)



        #calculate the loss for this batch
        p = np.transpose(self.y_for_this_batch) #doing transpose to make dimensions compatible

        loss = -(np.dot(p,np.log(self.sigmoid_layer_op)) + np.dot((1-p),(np.log(1-self.sigmoid_layer_op))))

        print(" y y cap Z1 loss ",self.y_for_this_batch,self.sigmoid_layer_op,self.hidden_layer_output, loss)
        #time.sleep(1000)



        #-----------------------------Now do backward pass









        #calculate the grad output to send to the backward fucntion of sigmod cross entropy



        #-------------------call backward function to ge dE by dz2
        #use that to calculate dZ2 by dW2
        print("here 1",(self.y_for_this_batch/self.sigmoid_layer_op))
        print("here ",(1-self.y_for_this_batch)/(1-self.sigmoid_layer_op))

        dE_by_dy_cap = -(np.divide((self.y_for_this_batch),(self.sigmoid_layer_op)) - (np.divide((1-self.y_for_this_batch),(1-self.sigmoid_layer_op))))

        print("dE_by_dy_cap",dE_by_dy_cap)
        print("************")
        direction_updated_weights_2, direction_updated_bias_2, backward_return_for_sencond_linear_transform = self.sigmoid_object_layer.backward(input, self.y_for_this_batch, self.sigmoid_layer_op, dE_by_dy_cap, learning_rate,direction, momentum, l2_penalty,self.linear_transform_object_for_sigmoid.weights, self.linear_transform_object_for_sigmoid.bias)



        print(" backward returns ",direction_updated_weights_2, direction_updated_bias_2, backward_return_for_sencond_linear_transform)




        #time.sleep(1000)






        #do similarly for W1 and b1
#------------------------------------continue from here
        #grad_output_for_lin_tr = np.dot(grad_output_for_W2 ,






        #call sigmoid backward that calls linear backward





        #return self.sigmoid_layer_op



	# INSERT CODE for training the network





    def evaluate(self, y_cap, y):

        print("y cap and y for loss ", y_cap, y)
        p = np.transpose(y)

        loss = -(np.dot(p,np.log(y_cap)) + np.dot((1-p),(np.log(1-y_cap))))



        print("loss is  ",loss)

        return loss




    def weight_update(self, loss_for_the_batch, lr, direction_passed, momentum,l2_penalty):

        print("y truth",self.y_for_this_batch,"y cap",self.y_cap_for_this_batch)



        x1 = np.dot(self.y_for_this_batch, np.transpose(( 1- self.y_for_this_batch)))
        x2 = (np.dot((self.y_cap_for_this_batch),np.transpose((1-self.y_cap_for_this_batch))))
        a = x1/x2


        b = np.dot(self.y_cap_for_this_batch, np.transpose((1 - self.y_cap_for_this_batch)))

        dE_by_dW2 = np.dot( np.dot(a,b),self.hidden_layer_output)




#         print("size ",x1.size,x2.size,a.size,b.size,dE_by_db2.size,dE_by_dW2.size)
#         print("a, b, grad ",a,b,dE_by_db2, self.second_layer_bias)

        #The dimension of dE by dW2 is 10 by 2
        #10 examples taken in a batch so each of them give a unique feedback for both the weights in W2 for the 2 input weights
        #to the sigmoid node
        #we need to sum them up and take an average to do the update



        dE_by_dW2 = (dE_by_dW2.sum(axis = 0))/10#--------------------------------dont forget to change batch size here

        #call the sigmoid unit to do the update on the weights by passing the necessary arguments


        #second layer weights updated here
        direction_passed, self.second_layer_weights =  self.sigmoid_object_for_backpass.backward(self.second_layer_weights, lr, direction_passed, momentum,l2_penalty, dE_by_dW2)

        #------------------second layer bias update


        dE_by_db2 = np.dot(a,b)






        #time.sleep(5)
        #NOW UPdate the first ;layer weights


        #dE_by_d_y_cap = y -

















	# INSERT CODE for testing the network
# ADD other operations and data entries in MLP if needed









if __name__ == '__main__':
    if sys.version_info[0] < 3:
        print("system version is less than 3")
        data = pickle.load(open('dataset_folder/cifar_2class_py2.p', 'rb'))

    else:
        #train_x, train_y, test_x, test_y
        data = pickle.load(open('../../dataset_folder/cifar_2class_py2.p', 'rb'), encoding='bytes')

    #print(data)
    #print(data[b'test_data'])


    train_x = np.array(data[b'train_data'])
    train_y = np.array(data[b'train_labels'])
    test_x = np.array(data[b'test_data'])
    test_y = np.array(data[b'test_labels'])



    def normalize(x):

        top = x - np.amin(x)
        bottom = np.amax(x)-np.amin(x)
        return(top/bottom)


    #for checking------------------remove paxi
    #print("dtype is ",train_x.dtype)
    train_x = train_x[...,0:1000]
    train_y = train_y[...,0:1000]
    test_x = test_x[...,0:1000]
    test_y = test_y[...,0:1000]



    #print(" minimum in x is ", np.amax(train_x))
#     print(" maximum in x is ", np.amax(train_x))
#     train_x = normalize(train_x)
#     print(" normalized x is ",train_x)
#     print(" maximum in x is ", np.amax(train_x))





    num_examples, input_dims = train_x.shape


	# INSERT YOUR CODE HERE


    #ask the user about the the number of hidden nodes hs/ she wants


    #inp = input("How many hidden nodes do you want?\n")
    #num_of_hidden_nodes = int(inp)


    num_of_hidden_nodes = 2
#     print("dimension of each example is ",input_dims)





    # YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES


    num_epochs = 10
    num_batches = 1000
    learning_rate = [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]
    inertia_of_momentum = [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]
    l2_penalty_factor = [0.0000001,0.0000003,0.000001,0.000003,0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01,0.03,1,3,10]
    print("Choose the corresponding index number for the learning rate you want to use")
    #lr = int(input("[0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]"))
    lr =5
    print("Choose the corresponding index number for the inertia of momentum you want to use")
    #iner = int(input("[0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]"))
    iner = 1
    print("Choose the corresponding index number for the L2 penalty factor you want to use")
    #penalty = int(input("[0.0000001,0.0000003,0.000001,0.000003,0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01,0.03,1,3,10]"))
    penalty = 1
    mlp = MLP(input_dims, num_of_hidden_nodes)



    epoch_num = 1
    for epoch in range(num_epochs):
        print(" Epoch is ",epoch_num)

	# INSERT YOUR CODE FOR EACH EPOCH HERE
        total_loss_for_epoch = 0.0
        direction = 0
        for b in range(num_batches):



            #print("num of examples num of batches ",num_examples, " ",num_batches)



            batch_start = int( (num_examples / num_batches) * b)
            batch_end = int((num_examples / num_batches)*(b +1))



#             print("batch start and end",batch_start," ",batch_end)
#             print("Chosen learning rate, inertia of momentum and l2 penalty factor are")
            #print(learning_rate[lr], inertia_of_momentum[iner], l2_penalty_factor[penalty])
            print("train size is  ",train_x.shape)





            mlp.x_for_this_batch = train_x[batch_start:batch_end,...]
            mlp.y_for_this_batch = train_y[batch_start:batch_end,...]
            mlp.y_cap_for_this_batch = mlp.train(mlp.x_for_this_batch,mlp.y_for_this_batch,int(learning_rate[lr]), inertia_of_momentum[iner], l2_penalty_factor[penalty])







            #print("y_cap_for_batch",mlp.y_cap_for_this_batch)

            loss = mlp.evaluate(mlp.y_cap_for_this_batch,mlp.y_for_this_batch )



            total_loss_for_epoch = total_loss_for_epoch + loss

            #call for weight updates,  need to pass ionly the loss

            mlp.weight_update(loss, learning_rate[lr], direction, inertia_of_momentum[iner], l2_penalty_factor[penalty])










            # INSERT YOUR CODE FOR EACH MINI_BATCH HERE
            # MAKE SURE TO UPDATE total_loss



            #print('\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(epoch + 1,b + 1,total_loss_for_epoch,),end='', )

            #after each mini bach update you want to update the momentum value
            sys.stdout.flush()
            # INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
        epoch_num +=1

        direction = 0
        # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy


        print()
        #do for each epoch this


        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
            train_loss,
            100. * train_accuracy,
        ))
        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
            test_loss,
            100. * test_accuracy,
        ))

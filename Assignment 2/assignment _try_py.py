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

        #print("First linear transform ", np.dot(np.transpose(self.weights),np.transpose(x) ) .shape,self.bias.shape)

        print("np.dot(np.transpose(self.weights),np.transpose(x) )",np.dot(np.transpose(self.weights),np.transpose(x) ).shape)
        print("self.bias",self.bias.shape)
        batch_linear_summation = np.dot(np.transpose(self.weights),np.transpose(x) ) + self.bias


        #+ np.full()self.bias


        return (batch_linear_summation)#,self.relu_object.forward(batch_linear_summation_without_relu,y))

    def forward_2(self,x):

        #print("shapes of x, self.weights, bias", self.weights.shape,x.shape,self.bias.shape)
        batch_linear_summation_without_sigmoid = np.dot(np.transpose(self.weights),x)+self.bias
        #print("batch_linear_summation_without_sigmoid",batch_linear_summation_without_sigmoid.shape)
        return (batch_linear_summation_without_sigmoid)



    def backward(self, input, grad_output,learning_rate,momentum,l2_penalty):
        #print("linear transform 2 ko backward bhitra ",np.transpose(grad_output).shape, np.transpose(self.weights).shape) #dE by dop dus ota hunuparxa for each example

        dE_by_dZ1 = np.dot(np.transpose(grad_output), np.transpose(self.weights))
        #print("dE_by_dZ1 shape",dE_by_dZ1.shape)

        return dE_by_dZ1





        print()
	# DEFINE backward function


class ReLU(object):
    def __init__(self):
        print()

    def forward(self, x):
	# DEFINE forward function
        #print("Relu Input is ",x)
        relu_output = np.maximum(0,x)
        #print("relu_output is",relu_output)
        return relu_output


    def backward( self, input, grad_output):

        #print("Relu backeard bhitra ", input, grad_output)
        dZ1_by_dA1 = input
        print("here",dZ1_by_dA1.dtype)

        np.where(dZ1_by_dA1 == 0, np.random.uniform(0.00001,1), dZ1_by_dA1)
        np.where(dZ1_by_dA1 < 0, 0, dZ1_by_dA1)
        np.where(dZ1_by_dA1 > 0, 1, dZ1_by_dA1)


        #print("relu output backward",dZ1_by_dA1)


        return(dZ1_by_dA1)


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

        #print("true and predicted",(true_output).shape,predicted_output.shape)
        dE_by_dA2 = np.transpose(true_output)-predicted_output
        #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",dE_by_dA2.shape)
        return(dE_by_dA2 )

        # print("#here hai",dE_by_dA2,l2_penalty, second_layer_wts)
        # direction_updated_weights_2 = momentum * direction - np.dot(learning_rate, ( dE_by_dA2 + np.dot(l2_penalty, second_layer_wts )))
        #
        # direction_updated_bias_2 = momentum * direction - np.dot(learning_rate, ( dE_by_dA2 + np.dot(l2_penalty, second_layer_bias )))
        #
        # print("direction updated orr weight update is ",direction_updated_weights_2)
        # print("bias direction updated orr bias update is ",direction_updated_bias_2)
        #
        #
        # second_layer_wts = second_layer_wts - direction_updated_weights_2
        # second_layer_bias = second_layer_bias - direction_updated_bias_2




        #-----------------------------------------------
        #call the previous layer for updates on the previous layer weights

        #backward_return_for_sencond_linear_transform = self.linear_transform_object_for_sigmoid.backward(self,grad_output, learning_rate,direction, momentum, l2_penalty,second_layer_wts,second_layer_bias)













# This is a class for the Multilayer perceptron




class MLP(object):

    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network

        self.input_dims = input_dims
        self.hidden_units = hidden_units
        #print("shapes here ererer ",self.input_dims,self.hidden_units)


        #np.full((2,2),(1,2)))
        self.linear_transform_object_first = LinearTransform(np.random.rand(input_dims, hidden_units),np.full((num_of_hidden_nodes,int(num_examples_per_batch)),np.random.rand(num_of_hidden_nodes,1)))#-----------------------MANUALLY done
        # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        # print(self.linear_transform_object_first.weights.shape)
        # print(self.linear_transform_object_first.bias.shape)
        self.relu_layer = ReLU()

        #print("np.full((1,1),np.random.rand(1))",np.full((1,1),np.random.rand(1)))
        #self.linear_transform_object_second = LinearTransform(np.random.rand(hidden_units, 1), np.full((1,1),np.random.rand(1)))
        self.linear_transform_object_second = LinearTransform(np.random.rand(hidden_units, 1), np.full((1,int(num_examples_per_batch)),np.random.rand(1)))

        #print(self.linear_transform_object_second.weights.shape)
        #print(self.linear_transform_object_second.bias.shape)
        self.sigmoid_object_layer = SigmoidCrossEntropy()




        #------------------------------------
        self.x_for_this_batch = []
        self.y_for_this_batch = []
        self.y_cap_for_this_batch = []


        self.Z1 = [] #Relu through aayeko xa sigmoid of A1
        self.sigmoid_output = []#Sigmoid through aayeko is Relu of A2, equals y cap for this batch

        self.A1 = [] #A1
        self.sigmoid_layer_op_before_sigmoid = []#A2
        #------------------


    def train(self,x_batch,y_batch,learning_rate,momentum,l2_penalty):

        self.x_for_this_batch = np.array(x_batch)
        self.y_for_this_batch = np.array(y_batch)
        #print("within train x_batch, y_batch",x_batch,y_batch)
        #print("x batch and y batch ",len(x_batch),len(y_batch))


        ###########------------First FOrward Pass and then backward pass---------




        #print("Sent X for first linear transform ",x_batch.shape)
        self.A1= np.array(self.linear_transform_object_first.forward(x_batch))
        #print("A1 shape",self.A1.shape)
        self.Z1 = np.array(self.relu_layer.forward(self.A1))
        #print("Z1 shape",self.Z1.shape)


        self.A2 = np.array(self.linear_transform_object_second.forward_2(self.Z1))
        #print("A2 shape",self.A2.shape)

        self.y_cap = np.array(self.sigmoid_object_layer.forward(self.A2))
        #print("y_cap shape",self.y_cap)


        #avoid overflow and underflow

        self.y_cap = np.where(self.y_cap == 0, 0.00000001, self.y_cap)


        self.y_cap = np.where(self.y_cap == 1, 1 - 0.00000001, self.y_cap)


        #calculate the loss for this batch
        y_transposed = np.transpose(self.y_for_this_batch) #doing transpose to make dimensions compatible
        y_cap_transposed = np.transpose(self.y_cap)
        #print("y ransposed ",y_transposed.shape)
        #print("y_cap_transposed",y_cap_transposed.shape)
        loss = -(np.dot(y_transposed, np.log(y_cap_transposed)) + np.dot((1 - y_transposed), (np.log(1 - y_cap_transposed))))

        #print(" LOSS  ", loss)
        #time.sleep(1000)



        #-----------------------------Now do backward pass




        #******************Check dimension of each one of them
        # print(" Dimensions of X1, A1",self.x_for_this_batch.shape)
        # print("Dimensions of Z1, A2",self.Z1.shape,self.A2.shape)




        #calculate the grad output to send to the backward fucntion of sigmod cross entropy



        #print("(self.y_for_this_batch-self.y_cap)",self.y_for_this_batch,self.y_cap)



        #dE_by_dy_cap = -(np.divide((self.y_for_this_batch), (self.y_cap)) - (np.divide((1 - self.y_for_this_batch), (1 - self.y_cap))))

        dE_by_dy_cap = -(np.divide((np.transpose(self.y_for_this_batch)-self.y_cap),(np.multiply(self.y_cap,(1-self.y_cap)))))




        #print("************ dE_by_dy_cap",dE_by_dy_cap.shape)

        dE_by_dA2 = self.sigmoid_object_layer.backward(input, self.y_for_this_batch, self.y_cap, dE_by_dy_cap, learning_rate, direction_for_W2, momentum, l2_penalty, self.linear_transform_object_second.weights, self.linear_transform_object_second.bias)


        #print("dE_by_dA2 shape ",dE_by_dA2.shape)

        #direction_updated_weights_2, direction_updated_bias_2, backward_return_for_sencond_linear_transform = self.sigmoid_object_layer.backward(input, self.y_for_this_batch, self.sigmoid_layer_op, dE_by_dy_cap, learning_rate,direction, momentum, l2_penalty,self.linear_transform_object_for_sigmoid.weights, self.linear_transform_object_for_sigmoid.bias)




        #print(" Sigmoid Cross Entropy backward returns ",dE_by_dA2.shape)

        #print("Shape of W2 is ",self.linear_transform_object_second.weights)
        dE_by_dW2 = np.dot(dE_by_dA2,np.transpose(self.Z1)) / num_examples_per_batch #averaging the update

        #dE_by_db2 = np.sum(dE_by_dA2,axis=1)/num_examples_per_batch
        dE_by_db2 = (dE_by_dA2)/num_examples_per_batch

        #print("de dE_by_db2 dw2 ",dE_by_db2.shape)


        dE_by_dZ1 = self.linear_transform_object_second.backward(self.Z1, dE_by_dA2, learning_rate, momentum, l2_penalty)


        #print("Linear Transform for Sigmoid backward returns ", dE_by_dZ1)



        #print("check size ",self.A1.shape,dE_by_dZ1.shape)

        dE_by_dA1 = self.relu_layer.backward(self.A1, dE_by_dZ1) #should return dE by dA1


        dE_by_dW1 = np.dot(dE_by_dA1, self.x_for_this_batch)/num_examples_per_batch

        #dE_by_db1 = (np.sum(dE_by_dA1,axis=1))/num_examples_per_batch
        dE_by_db1 = ((dE_by_dA1))/num_examples_per_batch
        #print("dE_by_db1",dE_by_db1.shape)

        #**********************************************Unit test all values may be?




        #*********************************Update both the weights simultaneopusly at the end





        print("\n\n\nFollowing are the dimensions of all the components")


        print("x_for_this_batch shape",self.x_for_this_batch.shape)
        print("\n\ny for this batch",self.y_for_this_batch.shape)
        print("\nA1",self.A1.shape)
        print("\nZ1",self.Z1.shape)
        print("\nA2",self.A2.shape)
        print("\nZ2",self.y_cap.shape)
        print("\ny transpose ",y_transposed.shape)
        print("\ny cap transpose ",y_cap_transposed.shape)
        print("\ndE+by_dy_cap ",dE_by_dy_cap.shape)
        print("\ndE_by_dA2 ",dE_by_dA2.shape)
        print("\ndE_by_dW2 ",dE_by_dW2.shape)
        #print("\nsecpn layer weights",mlp.linear_transform_object_second.weights.shape)
        print("\ndE_by_db2 ",dE_by_db2.shape)
        print("\ndE_by_dZ1 ",dE_by_dZ1.shape)
        print("\ndE_by_dA1 ",dE_by_dA1.shape)
        print("\ndE_by_dW1 ",dE_by_dW1.shape)
        print("dE_by_db1 ",dE_by_db1.shape)
        print("\nThe loss is ",loss)

        return(loss, dE_by_dW2,dE_by_db2,dE_by_dW1,dE_by_db1)












	# INSERT CODE for training the network





    def evaluate(self,x_data,y_data):


        print("\nInside Evaluate() function ")
        time.sleep(1)
        x_for_this_batch = np.array(x_data)
        y_for_this_batch = np.array(y_data)

        loss_val = 0

        for b in range(num_batches):


            print("BATCH NUMBER: ", b)
            print("num examples ",num_examples)
            #time.sleep(1)
            #print("num of examples num of batches ",num_examples, " ",num_batches)



            batch_start = int( (num_examples / num_batches) * b)
            batch_end = int((num_examples / num_batches)*(b +1))

            x_batch = x_for_this_batch[batch_start:batch_end,...]
            y_batch = y_for_this_batch[batch_start:batch_end,...]



            A1= np.array(self.linear_transform_object_first.forward(x_batch))
            #print("A1 shape",self.A1.shape)
            Z1 = np.array(self.relu_layer.forward(A1))
            #print("Z1 shape",self.Z1.shape)


            A2 = np.array(self.linear_transform_object_second.forward_2(Z1))
            #print("A2 shape",self.A2.shape)

            y_cap = np.array(self.sigmoid_object_layer.forward(A2))
            #print("y_cap shape",self.y_cap)


            #avoid overflow and underflow

            y_cap = np.where(y_cap == 0, 0.00000001, y_cap)


            y_cap = np.where(y_cap == 1, 1 - 0.00000001, y_cap)


            #calculate the loss for this batch
            y_transposed = np.transpose(y_batch) #doing transpose to make dimensions compatible
            y_cap_transposed = np.transpose(y_cap)
            #print("y ransposed ",y_transposed.shape)
            #print("y_cap_transposed",y_cap_transposed.shape)
            loss_val = -(np.dot(y_transposed, np.log(y_cap_transposed)) + np.dot((1 - y_transposed), (np.log(1 - y_cap_transposed))))

            print("Loss value is ",loss_val)
            #time.sleep(3)
            loss_val += loss_val




        return loss_val




    def weight_update(self, loss, dE_by_dW2,dE_by_db2,dE_by_dW1,dE_by_db1,learning_rate, direction_w2,direction_W1,direction_b2,direction_b1, inertia_of_momentum, l2_penalty_factor):




        new_direction_for_W2 = inertia_of_momentum * direction_w2 - np.dot(learning_rate, (( np.transpose(dE_by_dW2) + np.dot(l2_penalty_factor, self.linear_transform_object_second.weights ))))



        self.linear_transform_object_second.weights += new_direction_for_W2

        #print("old direction b2",direction_b2.shape)

        #print("old b2",self.linear_transform_object_second.bias.shape)
        #print("yo k ho ta ",( np.array(dE_by_db2) + np.dot(l2_penalty_factor, self.linear_transform_object_second.bias )))
        new_direction_for_b2 = inertia_of_momentum * direction_b2 - np.dot(learning_rate, (( np.array(dE_by_db2) + np.dot(l2_penalty_factor, self.linear_transform_object_second.bias ))))
        #print("to add to  direction b2",new_direction_for_b2.shape)

        self.linear_transform_object_second.bias += new_direction_for_b2

        #print("new b2 shape ",self.linear_transform_object_second.bias.shape)

        new_direction_for_W1 = inertia_of_momentum * direction_W1 - np.dot(learning_rate, (( np.transpose(dE_by_dW1) + np.dot(l2_penalty_factor, self.linear_transform_object_first.weights ))))
        self.linear_transform_object_first.weights += new_direction_for_W1


        #print("old b1 direction",direction_b1.shape)

        #print("old b1",self.linear_transform_object_first.bias.shape)


        new_direction_for_b1 = inertia_of_momentum *(direction_b1) - np.dot(learning_rate, (( np.array(dE_by_db1) + np.dot(l2_penalty_factor, (self.linear_transform_object_first.bias )))))

        #print("to add to b1 ",new_direction_for_b1.shape)


        self.linear_transform_object_first.bias += new_direction_for_b1        #= np.sum(np.array(self.linear_transform_object_first.bias), new_value_for_bias1_update)
        #print("b1 shape after update", self.linear_transform_object_first.bias.shape)

        #return (self.linear_transform_object_second.weights,self.linear_transform_object_second.bias, self.linear_transform_object_first,self.linear_transform_object_first)
















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
    #print("train x shape original is  ",train_x.shape)
    train_y = np.array(data[b'train_labels'])
    test_x = np.array(data[b'test_data'])
    test_y = np.array(data[b'test_labels'])



    def normalize(x):

        top = x - np.amin(x)
        bottom = np.amax(x)-np.amin(x)
        return(top/bottom)


    #for checking------------------remove paxi
    #print("dtype is ",train_x.dtype)
    # train_x = train_x[0:5000,...]
    # train_y = train_y[0:5000,...]
    # test_x = test_x[0:5000,...]
    # test_y = test_y[0:5000,...]



    #print(" minimum in x is ", np.amax(train_x))
#     print(" maximum in x is ", np.amax(train_x))
    train_x = normalize(train_x)
    #print(" normalized x is ",train_x)
#     print(" maximum in x is ", np.amax(train_x))





    #print(" All shapre are ",train_x.shape,train_y.shape,test_x.shape,test_y.shape)
    num_examples, input_dims = train_x.shape
    #print(" Number of examples and iunput dimensions are ",num_examples, input_dims)
    num_of_hidden_nodes = 5
  # YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES


    num_epochs = 10
    num_batches = 100

    #print("num_examples is ",num_examples)
    num_examples_per_batch = num_examples / num_batches
    learning_rate = [0.0000001,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]
    inertia_of_momentum = [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]
    l2_penalty_factor = [0.0000001,0.0000003,0.000001,0.000003,0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01,0.03,1,3,10]
    print("Choose the corresponding index number for the learning rate you want to use")
    #lr = int(input("[0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]"))
    lr =1
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
        direction_for_W2 = np.zeros(mlp.linear_transform_object_second.weights.shape)
        direction_for_W1 = np.zeros(mlp.linear_transform_object_first.weights.shape)
        direction_for_b2 = np.zeros(mlp.linear_transform_object_second.bias.shape)
        #print("shape linear_transform_object_first.bias ",mlp.linear_transform_object_first.bias.shape)
        direction_for_b1 = np.zeros(mlp.linear_transform_object_first.bias.shape)

        #print("directions ",direction_for_b2.shape)
        for b in range(num_batches):


            print("BATCH NUMBER: ", b)
            #print("num of examples num of batches ",num_examples, " ",num_batches)



            batch_start = int( (num_examples / num_batches) * b)
            batch_end = int((num_examples / num_batches)*(b +1))



#             print("batch start and end",batch_start," ",batch_end)
#             print("Chosen learning rate, inertia of momentum and l2 penalty factor are")
            #print(learning_rate[lr], inertia_of_momentum[iner], l2_penalty_factor[penalty])





            #print("mlp.x_for_this_batch ",(train_x[batch_start:batch_end,...]).shape)
            mlp.x_for_this_batch = train_x[batch_start:batch_end,...]
            mlp.y_for_this_batch = train_y[batch_start:batch_end,...]

            #__________________________________________________________________TRAINING_____________________________________________________________________________
            loss, dE_by_dW2,dE_by_db2,dE_by_dW1,dE_by_db1 = mlp.train(mlp.x_for_this_batch,mlp.y_for_this_batch,int(learning_rate[lr]), inertia_of_momentum[iner], l2_penalty_factor[penalty])

            #print(" Initially W2 was and b2 was ", mlp.linear_transform_object_second.weights, mlp.linear_transform_object_second.bias)

            #___________________________________________________________________WEIGHT UPDATE_________________________________________________________________________________
            mlp.weight_update(loss, dE_by_dW2, dE_by_db2, dE_by_dW1, dE_by_db1, learning_rate[lr], direction_for_W2, direction_for_W1, direction_for_b2, direction_for_b1, inertia_of_momentum[iner], l2_penalty_factor[penalty])
            #print(" After the update, W2 was and b2 was ", mlp.linear_transform_object_second.weights, mlp.linear_transform_object_second.bias)

            total_loss_for_epoch = total_loss_for_epoch + loss













            # INSERT YOUR CODE FOR EACH MINI_BATCH HERE
            # MAKE SURE TO UPDATE total_loss



            #print('\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(epoch + 1,b + 1,total_loss_for_epoch,),end='', )

            #after each mini bach update you want to update the momentum value
            sys.stdout.flush()
            # INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
        epoch_num +=1

        direction = 0
        # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy



        #do for each epoch this

        #each epeoch ko end ma training ra testing duitai data ma test garne aaba

        #tesko lagi evaluate function banai


        #------------------------------------Lets test the network------------------------------------------------------------------------------
        print("Evaluating loss for training data")
        time.sleep(1)

        training_loss_for_this_epoch = mlp.evaluate(train_x,train_y)
        print("Evaluating loss for testing data")
        time.sleep(1)
        testing_loss_for_this_epoch = mlp.evaluate(test_x,test_y)

        print("Training loss for this epoch", training_loss_for_this_epoch)

        time.sleep(1)
        print("Testing loss for this epoch ",testing_loss_for_this_epoch)
        time.sleep(1)


        #print('Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(training_loss_for_this_epoch,100. * train_accuracy,))
        # print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(testing_loss_for_this_epoch, 100. * test_accuracy,))

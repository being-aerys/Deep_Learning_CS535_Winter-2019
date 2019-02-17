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
import matplotlib.pyplot as plt



class LinearTransform(object):

    def __init__(self, W, b):

        self.weights = W
        #print("Random weights ",self.weights)

        self.bias = b
        #print("Random bias ",self.bias)
        #time.sleep(5)



    def forward(self, x):

        #Never do this because it does element-wise and this gives an error
        #batch_linear_summation = x * self.weights# + self.bias


        batch_linear_summation = np.dot(np.transpose(self.weights),np.transpose(x) ) + self.bias
        #print("ss",batch_linear_summation)
        #time.sleep(5)

        #+ np.full()self.bias


        return (batch_linear_summation)#,self.relu_object.forward(batch_linear_summation_without_relu,y))

    def forward_2(self,x):

        #print("shapes of x, self.weights, bias", self.weights.shape,x.shape,self.bias.shape)
        #time.sleep(111)
        batch_linear_summation_without_sigmoid = np.dot(np.transpose(self.weights),x)+self.bias
        #print("batch_linear_summation_without_sigmoid",batch_linear_summation_without_sigmoid)
        #time.sleep(222)
        return (batch_linear_summation_without_sigmoid)



    def backward(self, input, grad_output,learning_rate,momentum,l2_penalty):
        #print("linear transform 2 ko backward bhitra ",np.transpose(grad_output).shape, np.transpose(self.weights).shape) #dE by dop dus ota hunuparxa for each example
        #print("here inside sha[pes ",np.transpose(grad_output).shape, np.transpose(self.weights).shape)


        dE_by_dZ1_to_return = np.dot(np.transpose(grad_output), np.transpose(self.weights))


        return dE_by_dZ1_to_return






	# DEFINE backward function


class ReLU(object):
    def __init__(self):
        print()

    def forward(self, x):
	# DEFINE forward function
        #print("Relu Input is ",x)

        relu_output = np.maximum(0,x)

        #print("relu_output is",relu_output)
        #time.sleep(1111)
        return relu_output


    def backward( self, A1_matrix, grad_output):

        #print("Relu backeard bhitra ", input, grad_output)

        #print("here",dZ1_by_dA1.dtype)

        #print("i",grad_output.shape)
        A1_matrix = A1_matrix.transpose()


        #print("np.multiply(grad_output,np.random.random_sample())",np.multiply(grad_output,np.random.random_sample()).shape)

        # np.where(input == 0, np.multiply(grad_output,np.random.random_sample()),grad_output)
        # np.where(input < 0, 0, grad_output)
        # np.where(input > 0, grad_output, grad_output)
        #
        #
        # #print("relu output backward",grad_output.shape)

        #a = np.heaviside(input, np.multiply(np.random.random_sample(), grad_output)) # no of examples * no of hidden nodes

        np.where(A1_matrix < 0, 0, A1_matrix)
        np.where(A1_matrix > 0, 1, A1_matrix)#---------------------------------------ignoring inputs = 0 for now
        #np.where(A1_matrix == 0, np.multiply(np.random.random, A1_matrix),A1_matrix)
        dE1_by_dA1_to_retuen = np.multiply(grad_output, A1_matrix)


        return(dE1_by_dA1_to_retuen)


class SigmoidCrossEntropy(object):
    def __init__(self):
        print()
        self.input_values=[]
        #self.linear_transform_object_2 = LinearTransform(self.second_layer_weights,self.second_layer_bias)

    def forward(self, x):
#
        #self.input_values = x
        sigmoid =  (1.0 / (1 + np.exp(-x)))
        #print("sigmoid is ", sigmoid)
        return sigmoid

    def backward(self,input, true_output, predicted_output, grad_output, learning_rate,direction, momentum, l2_penalty,second_layer_wts, second_layer_bias ):

        #print("true and predicted",(true_output).shape,predicted_output.shape)
        dE_by_dA2 =predicted_output -  np.transpose(true_output)
        #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",dE_by_dA2.shape)
        return(dE_by_dA2 )


# This is a class for the Multilayer perceptron

class MLP(object):

    def __init__(self, input_dims, hidden_units):

        self.input_dims = input_dims
        self.hidden_units = hidden_units
        #print("shapes here ererer ",self.input_dims,self.hidden_units)




        self.linear_transform_object_first = LinearTransform(0.1 * np.random.randn(input_dims, hidden_units),0.1 * np.full((num_of_hidden_nodes,int(num_examples_per_batch)),np.random.random((num_of_hidden_nodes,1))))#-----------------------MANUALLY done

        self.relu_layer = ReLU()


        self.linear_transform_object_second = LinearTransform(0.1 * np.random.randn(hidden_units, 1), 0.1 * np.full((1,int(num_examples_per_batch)),np.random.random(1)))



        self.sigmoid_object_layer = SigmoidCrossEntropy()


        #------------------------------------
        self.x_for_this_batch = []
        self.y_for_this_batch = []
        #self.y_cap_for_this_batch = []



    def train(self,x_batch,y_unrolled,learning_rate,momentum,l2_penalty):

        self.x_for_this_batch = np.array(x_batch)
        self.y_for_this_batch = y_unrolled
        #print("a",self.y_for_this_batch)
        #time.sleep(111)




        self.A1= np.array(self.linear_transform_object_first.forward(x_batch))
        #print("A1 shape",self.A1)

        self.Z1 = np.array(self.relu_layer.forward(self.A1))
        #print("Z1 shape",self.Z1)
        #time.sleep(2222)

        self.A2 = np.array(self.linear_transform_object_second.forward_2(self.Z1))
        # print("A2 value in training ",self.A2)
        # print("W2 ",self.linear_transform_object_second.weights)
        self.y_cap = np.array(self.sigmoid_object_layer.forward(self.A2))
        #print("y_cap is",self.y_cap)
        #time.sleep(10)
        #y_transposed = np.transpose(self.y_for_this_batch) #doing transpose to make dimensions compatible
#-------------------------------------------------removing to chack if small or not
        # avoid overflow and underflow
        #
        #print("self y cap ",self.y_cap)

        self.y_cap = np.clip(self.y_cap,1e-12,1-(1e-12))
        # self.y_cap = np.where(self.y_cap == 0, 0.0000001, self.y_cap)
        #
        #
        #
        # self.y_cap = np.where(self.y_cap == 1, 1 - 0.0000001, self.y_cap)
        #

#--------------------------------------------------------------------------
        #calculate the loss for this batch


        #print("y_cap is ",np.log(self.y_cap+1e-9))

        #print("ss",(y_transposed*np.log(self.y_cap+1e-9)).shape)
        #time.sleep(222)
        #loss = -(np.sum(y_unrolled*np.log(self.y_cap+1e-9))+ np.sum((1 - y_unrolled * np.log((1 - self.y_cap+1e-9)))))/num_examples_per_batch
        #loss = - (np.dot(y_transposed,np.transpose(np.log(self.y_cap)))+ np.dot((1-y_transposed),np.transpose(np.log(1-self.y_cap))))
        #print(self.y_cap)
        loss = - (np.multiply(y_unrolled,np.transpose(np.log(self.y_cap)))+ np.multiply((1-y_unrolled),np.transpose(np.log(1-self.y_cap))))/num_examples_per_batch

        #print("loss ",loss)
        #time.sleep(1111)
        # first_part_of_loss = np.multiply(y_transposed, np.log(self.y_cap))
        #
        # #print("first ",first_part_of_loss.shape)
        # summed_loss_part_1 = np.sum(first_part_of_loss,axis=1)
        #
        # second_part_of_loss = np.multiply((1-y_transposed),np.log(1-self.y_cap))
        # summed_loss_part_2 = np.sum(second_part_of_loss,axis = 1)
        #
        # #time.sleep(22)
        # #loss = -(np.dot(y_transposed, np.log(y_cap_transposed)) + np.dot((1 - y_transposed), (np.log(1 - y_cap_transposed))))
        # loss = -(summed_loss_part_1 + summed_loss_part_2)/num_examples_per_batch


        #-----------------------------Now do backward pass

        #dE_by_dy_cap = -(np.divide((np.transpose(self.y_for_this_batch)-self.y_cap),(np.multiply(self.y_cap,(1-self.y_cap)))))

        dE_by_dA2 = self.sigmoid_object_layer.backward(input, self.y_for_this_batch, self.y_cap, 0, learning_rate, direction_for_W2, momentum, l2_penalty, self.linear_transform_object_second.weights, self.linear_transform_object_second.bias)



        # print("Shape of W2 is ",dE_by_dA2.shape,self.Z1.shape)
        # print(mlp.linear_transform_object_second.weights.shape)
        #time.sleep(111)
        dE_by_dW2 = (np.dot(dE_by_dA2,np.transpose(self.Z1)))/ num_examples_per_batch # divide by batch size----------------This is correct
        #print("shapesaaaa ",dE_by_dW2.shape)
        #



        #dE_by_db2 = np.sum(dE_by_dA2,axis=1)/num_examples_per_batch
        dE_by_db2 = (dE_by_dA2)/ num_examples_per_batch

        #print("de dE_by_db2 dw2 ",dE_by_db2.shape)
        #time.sleep(1111)

        dE_by_dZ1_not_summed = self.linear_transform_object_second.backward(self.Z1, dE_by_dA2, learning_rate, momentum, l2_penalty)
        #dE_by_dZ1 = np.reshape(np.sum(dE_by_dZ1_not_summed,axis=0),(1,num_hidden_nodes))



        #print("check size ",self.A1.shape,dE_by_dZ1.shape)

        dE_by_dA1_unsummed = self.relu_layer.backward(self.A1, dE_by_dZ1_not_summed) #should return dE by dA1

        # print("dimension ",dE_by_dA1_unsummed)
        # time.sleep(111)



        #
        # #print("dE by dA1, each row is for one example",dE_by_dA1_unsummed.shape)
        #
        # #print("input is  ",x_batch.shape)
        # t = time.time()
        # #print(int(input_dims))
        # summed_value = np.zeros([int(num_of_hidden_nodes),int(input_dims)])
        # for i in range(int(num_examples_per_batch)):
        #     column = np.reshape(dE_by_dA1_unsummed[i],(num_of_hidden_nodes,1))
        #
        #     repeated = np.repeat((column),input_dims,axis=1)
        #     #print("check ",repeated.shape)
        #
        #     summed_value = summed_value + np.multiply(repeated,x_batch[int(i),...])
        #     #print("example single ko shape aayo ta",summed_value.shape)
        #
        #
        # #print("time is ",time.time()-t)

        # dE_by_dW1 = summed_value/num_examples_per_batch

        #dE_by_db1 = np.sum(dE_by_dA1_unsummed,axis = 0)/num_examples_per_batch

        #print(dE_by_dA1_unsummed.shape, self.x_for_this_batch.shape)
        dE_by_dW1 = np.dot(np.transpose(dE_by_dA1_unsummed), self.x_for_this_batch)/ num_examples_per_batch

        #print(dE_by_dW1.shape)
        #time.sleep(1111)
        dE_by_db1 = np.transpose(dE_by_dA1_unsummed)/ num_examples_per_batch#-------------------------------This is correct

        #time.sleep(111)

        #print("ss",dE_by_dA1_unsummed[0].shape)
        #print(self.x_for_this_batch.shape)




        #dE_by_W1 = self.linear_transform_object_first.backward(self.)


        #dE_by_dW1 = np.dot(dE_by_dA1_unsummed, self.x_for_this_batch)

        #dE_by_db1 = ((dE_by_dA1_unsummed))
        #print("dE_by_db1",dE_by_db1.shape)



        #*********************************Update both the weights simultaneopusly at the end


        # 0 1 1 1 1 0 0 1 1 1 0 0 1 1 0 1 0 1 0 0 0 0 1 0 1 1 0 1 1 1 0 0 0 0 1 1
        # 0 1 1 1 1 0 0 1 1 1 0 0 1 1 0 1 0 1 0 0 0 0 1 0 1 1 0 1 1 1 0 0 0 0 1 1


        # print("\n\n\nFollowing are the dimensions of all the components")
        # print("x_for_this_batch shape",self.x_for_this_batch.shape)
        # print("\n\ny for this batch",self.y_for_this_batch.shape)
        # print("\nA1",self.A1)
        # print("\nZ1",self.Z1)
        # print("\nA2",self.A2)
        # print("\nZ2",self.y_cap)
        # print("\ny transpose ",y_transposed.shape)
        # # print("\ny cap transpose ",y_cap_transposed.shape)
        # # print("\ndE+by_dy_cap ",dE_by_dy_cap.shape)
        # print("\ndE_by_dA2 ",dE_by_dA2.shape)
        # print("\ndE_by_dW2 ",dE_by_dW2.shape)
        # #print("\nsecpn layer weights",mlp.linear_transform_object_second.weights.shape)
        # print("\ndE_by_db2 ",dE_by_db2.shape)
        # print("\ndE_by_dZ1 ",dE_by_dZ1_not_summed.shape)
        # print("\ndE_by_dA1 ",dE_by_dA1_unsummed.shape)
        # print("\ndE_by_dW1 ",dE_by_dW1.shape)
        # print("dE_by_db1 ",dE_by_db1.shape)
        # print("\nThe loss is ",loss)
        # print(" W1 is ",mlp.linear_transform_object_first.weights.shape)
        # print(("W2 is ",mlp.linear_transform_object_second.weights.shape))
        #time.sleep(10000)

        return(loss, 151 , dE_by_dW2,dE_by_db2,dE_by_dW1,dE_by_db1)












	# INSERT CODE for training the network





    def evaluate(self,x_data,y_data,check_tr_or_test):

        if check_tr_or_test == 0:  #training data


            x_for_this_batch_tr = np.array(x_data)
            y_for_this_batch_tr = np.array(y_data)

            loss_val = 0
            total_acc_for_whole_data = 0
            total_examples_inside_evaluate = 0
            #print("W2 weights  ",self.linear_transform_object_second.weights)
            time.sleep(1)

            for b in range(num_batches):
                #print("Inside Evaluate : Batch Number is  ", b)
                #print("tOTAL EXAMPLES obtained IN EVALUTE TILL NOW ", total_examples_inside_evaluate)


                #print("num examples ",num_examples)
                #time.sleep(1)
                #print("num of examples num of batches ",num_examples, " ",num_batches)
                #print("for this data total loss till now is ",loss_val)



                batch_start = int( (num_examples / num_batches) * b)
                batch_end = int((num_examples / num_batches)*(b +1))
                #print("batch starts at ",batch_start)
                #print("for this batch batch start ",batch_start)
                #print("batch ebnd", batch_end)
                #time.sleep(3)


                x_batch = x_for_this_batch_tr[batch_start:batch_end,...]

                y_batch = y_for_this_batch_tr[batch_start:batch_end,...]
                #print("inside ealutate 1 batch size is  ",len(y_batch))
                #time.sleep(1)
                #total_examples_inside_evaluate = total_examples_inside_evaluate +

                total_examples_inside_evaluate = total_examples_inside_evaluate + len(y_batch)




                A1= np.array(self.linear_transform_object_first.forward(x_batch))
                #print("A1 shape",self.A1.shape)
                Z1 = np.array(self.relu_layer.forward(A1))
                #print("Z1 shape",self.Z1.shape)


                A2 = np.array(self.linear_transform_object_second.forward_2(Z1))
                #print("A2 shape",self.A2)
                #time.sleep(9888)
                y_cap_evaluate = np.array(self.sigmoid_object_layer.forward(A2))
                #print("y cap evaluate before adjusting ", y_cap_evaluate)
                #time.sleep(1111)
                y_cap_evaluate = np.where(y_cap_evaluate >0.5, 1, 0)
                #print("y cap in evaluate after adjusting is  ",y_cap_evaluate)



                y_transposed = np.transpose(y_batch) #doing transpose to make dimensions compatible
                #print(" y cap head ",y_cap_evaluate)
                #print("y ",y_batch)
                #print("y_cap ",y_cap)
                #time.sleep(5)

                #y_cap_transposed = np.transpose(y_cap_evaluate)
                #print("y ransposed ",y_transposed.shape)
                #print("y_cap_transposed",y_cap_transposed.shape)


                #----------------calculat eaccuracy
                #print(y_transposed)
                #print("\n\nIn evaluate for training data",y_cap_evaluate)
                accuracy = np.sum (y_transposed == y_cap_evaluate)
                #print("accuracy in training evaluation is ",accuracy)
                #time.sleep(3)



                #loss_val = -(np.dot(y_transposed, np.log(y_cap_transposed)) + np.dot((1 - y_transposed), (np.log(1 - y_cap_transposed))))

                #print("Loss value is ",loss_val)
                #time.sleep(3)
                #print("old training data accuracy for whole is  ",total_acc_for_whole_data)
                total_acc_for_whole_data += accuracy
                #print("New accuracy? ", total_acc_for_whole_data)
                #time.sleep(3)



            #print("acc val for whole data is ", total_acc_for_whole_data)



            return total_acc_for_whole_data #yesto nagare 2-dimensional array return garxa feri



        else:   #testing data
            #print("no of batch for testing", int(2000/int(num_examples_per_batch)))
            x_for_this_batch_test = np.array(x_data)
            y_for_this_batch_test = np.array(y_data)


            total_acc_for_whole_data_test = 0
            total_examples_inside_evaluate_test = 0


            for q in range(int(2000/int(num_examples_per_batch))):
                #print("Inside Evaluate : Batch Number is  ", b)
                #print("tOTAL EXAMPLES obtained IN EVALUTE TILL NOW ", total_examples_inside_evaluate)


                #print("num examples ",num_examples)
                #time.sleep(1)
                #print("num of examples num of batches ",num_examples, " ",num_batches)
                #print("for this data total loss till now is ",loss_val)



                batch_start = int( (num_examples / num_batches) * q)
                batch_end = int((num_examples / num_batches)*(q +1))

                #print("for this batch batch start ",batch_start)
                #print("batch ebnd", batch_end)
                #time.sleep(3)


                x_batch = x_for_this_batch_test[batch_start:batch_end,...]

                y_batch = y_for_this_batch_test[batch_start:batch_end,...]
                #print("inside ealutate 1 batch size is  ",len(y_batch))
                #time.sleep(1)
                #total_examples_inside_evaluate = total_examples_inside_evaluate +

                total_examples_inside_evaluate_test = total_examples_inside_evaluate_test + len(y_batch)




                A1= np.array(self.linear_transform_object_first.forward(x_batch))
                #print("A1 shape",self.A1.shape)
                Z1 = np.array(self.relu_layer.forward(A1))
                #print("Z1 shape",self.Z1.shape)


                A2 = np.array(self.linear_transform_object_second.forward_2(Z1))
                #print("A2 shape",self.A2)
                #time.sleep(9888)
                y_cap_evaluate_test = np.array(self.sigmoid_object_layer.forward(A2))
                #print("y cap evaluate before adjusting for test ", y_cap_evaluate)
                #time.sleep(111)
                y_cap_evaluate_test = np.where(y_cap_evaluate_test >0.5, 1, 0)
                #print("y cap in evaluate test data  after adjusting is  ",y_cap_evaluate)
                #time.sleep(111)


                y_transposed_test = np.transpose(y_batch) #doing transpose to make dimensions compatible
                #print("y ",y_batch)
                #print("y_cap ",y_cap)
                #time.sleep(5)

                #y_cap_transposed = np.transpose(y_cap_evaluate)
                #print("y ransposed ",y_transposed.shape)
                #print("y_cap_transposed",y_cap_transposed.shape)


                #----------------calculat eaccuracy
                # print(y_transposed)
                #print("\n\nIn evaluate for testing data",y_cap_evaluate)
                accuracy_test = np.sum (y_transposed_test == y_cap_evaluate_test)
                #print("accuracy in test data is ",accuracy)
                #time.sleep(3)



                #loss_val = -(np.dot(y_transposed, np.log(y_cap_transposed)) + np.dot((1 - y_transposed), (np.log(1 - y_cap_transposed))))

                #print("Loss value is ",loss_val)
                #time.sleep(3)
                #print("old test accuracy for all test data is  ",total_acc_for_whole_data_test)
                total_acc_for_whole_data_test += accuracy_test
                #print("New test accuracy for all test data ", total_acc_for_whole_data_test)
                #time.sleep(3)



            #print("acc val for whole data is ", total_acc_for_whole_data)


            print(y_cap_evaluate_test)
            return total_acc_for_whole_data_test #yesto nagare 2-dimensional array return garxa feri




    def weight_update(self, loss, dE_by_dW2,dE_by_db2,dE_by_dW1,dE_by_db1,learning_rate, direction_w2,direction_W1,direction_b2,direction_b1, inertia_of_momentum, l2_penalty_factor):




        direction_for_W2 = np.multiply(inertia_of_momentum , direction_w2) - np.multiply(learning_rate, (( np.transpose(dE_by_dW2) + np.multiply(l2_penalty_factor, self.linear_transform_object_second.weights ))))


        #print(" update in the weights is ",direction_for_W2)
        # print(self.linear_transform_object_second.weights)
        # time.sleep(533)
        self.linear_transform_object_second.weights = self.linear_transform_object_second.weights + direction_for_W2
        #self.linear_transform_object_second.weights-=learning_rate * np.transpose(dE_by_dW2)

        #print(" Norm of the gradient 2nd W ",np.linalg.norm(np.array(self.linear_transform_object_second.weights)))
        #time.sleep(3)

        #print("self.linear_transform_object_second.weights",self.linear_transform_object_second.weights)

        #print("old b2",self.linear_transform_object_second.bias.shape)
        #print("yo k ho ta ",( np.array(dE_by_db2) + np.dot(l2_penalty_factor, self.linear_transform_object_second.bias )))
        #print(self.linear_transform_object_second.bias)
        #time.sleep(533)
        direction_for_b2 = np.multiply(inertia_of_momentum , direction_b2) - np.multiply(learning_rate, (( np.array(dE_by_db2) + np.multiply(l2_penalty_factor, self.linear_transform_object_second.bias ))))
        #print("to add to  direction b2",new_direction_for_b2.shape)

        self.linear_transform_object_second.bias = self.linear_transform_object_second.bias+direction_for_b2

        #print("new b2 shape ",self.linear_transform_object_second.bias.shape)

        direction_for_W1 = np.multiply(inertia_of_momentum , direction_W1) - np.dot(learning_rate, (( np.transpose(dE_by_dW1) + np.dot(l2_penalty_factor, self.linear_transform_object_first.weights ))))

        self.linear_transform_object_first.weights =  self.linear_transform_object_first.weights + direction_for_W1
        #print(self.linear_transform_object_first.weights.shape)
        #time.sleep(22)





        #---------------need to do the divifsion on summaitno of weights











        #self.linear_transform_object_first.weights-=learning_rate * np.transpose(dE_by_dW1)


        #print("old b1 direction",direction_b1.shape)

        #print("old b1",self.linear_transform_object_first.bias.shape)


        direction_for_b1 = np.multiply(inertia_of_momentum , direction_b1) - np.dot(learning_rate, (( np.array(dE_by_db1) + np.dot(l2_penalty_factor, (self.linear_transform_object_first.bias )))))

        #print("to add to b1 ",new_direction_for_b1.shape)


        self.linear_transform_object_first.bias = self.linear_transform_object_first.bias + direction_for_b1        #= np.sum(np.array(self.linear_transform_object_first.bias), new_value_for_bias1_update)
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


    train_x = np.array(data[b'train_data'])
    train_y = np.array(data[b'train_labels'])
    test_x = np.array(data[b'test_data'])
    test_y = np.array(data[b'test_labels'])





    # def normalize(x,min,max):
    #
    #     top = x - min
    #     bottom = max-min
    #     return(top/bottom)
    #


    train_x = train_x[0:10000,...]
    train_y = train_y[0:10000,...]
    test_x = test_x[0:10000,...]
    test_y = test_y[0:10000,...]
    #print(train_y)
    #time.sleep(222)






    train_x = (train_x - train_x.min(axis = 0)) / np.var(train_x,axis= 0) #taking normalized values, take max and min of each column
    test_x = (test_x - test_x.min(axis = 0)) / np.var(test_x,axis= 0)
    #print(train_x,test_x)
    #time.sleep(111)

    num_examples, input_dims = train_x.shape
    num_of_hidden_nodes = 100
    num_epochs = 500
    epoch_list_for_plot = []
    Training_accuracy_list = []
    Testing_accuracy_list = []
    num_batches = 100
    #num_batches_for_test_time = num_batches / (len(train_y)/len(test_y)) #

    num_examples_per_batch = num_examples / num_batches


    learning_rate = [1,0.00000001,0.001,0.002,0.003,0.01,0.03,0.1,0.3,1,3,10,20]
    inertia_of_momentum = [0,0.8,0.003,0.5,0.008,0.01,0.03,0.1,0.3,0.5,0.6,0.7,0.8,0.9,1,3,10]
    l2_penalty_factor = [0,0.001,0.0000001,0.0000003,0.000001,0.000003,0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01,0.03,1,3,10,20]
    #print("Choose the corresponding index number for the learning rate you want to use")
    #lr = int(input("[0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]"))
    lr =  0
    #print("Choose the corresponding index number for the inertia of momentum you want to use")
    #iner = int(input("[0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]"))
    iner = 0
    #print("Choose the corresponding index number for the L2 penalty factor you want to use")
    #penalty = int(input("[0.0000001,0.0000003,0.000001,0.000003,0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01,0.03,1,3,10]"))
    penalty = 0

    print(" num of hidden nodes: ",num_of_hidden_nodes)
    print("num of total epochs is ",num_epochs)
    print("num of examples per batch ",num_examples_per_batch)
    print("num of batches ",num_batches)
    print("num of epochs",num_epochs)
    print("learning rate is ",learning_rate[lr])
    print("momentum coefficient is ",inertia_of_momentum[iner])
    print("l2 penalty factor is ",l2_penalty_factor[penalty])


    mlp = MLP(input_dims, num_of_hidden_nodes)



    epoch_num = 1
    for epoch in range(num_epochs):

        print(" Epoch is ",epoch_num)
        print("y cap",mlp.linear_transform_object_second.weights)
        time.sleep(1)
        #time.sleep(1)

	# INSERT YOUR CODE FOR EACH EPOCH HERE
        total_loss_for_epoch = 0.0
        #total_accuracy_for_the_epoch = 0

        #print("shape linear_transform_object_first.bias ",mlp.linear_transform_object_first.bias.shape)
        direction_for_W2 = np.zeros(mlp.linear_transform_object_second.weights.shape)
        direction_for_W1 = np.zeros(mlp.linear_transform_object_first.weights.shape)
        direction_for_b2 = np.zeros(mlp.linear_transform_object_second.bias.shape)
        direction_for_b1 = np.zeros(mlp.linear_transform_object_first.bias.shape)


        for b in range(num_batches):


            #print("Epoch No", epoch_num,"BATCH NUMBER: ", b)

            batch_start = int( (num_examples / num_batches) * b)
            batch_end = int((num_examples / num_batches)*(b +1))
            #print("batch start is ",batch_start)


            mlp.x_for_this_batch = train_x[batch_start:batch_end,...]
            mlp.y_for_this_batch = train_y[batch_start:batch_end,...]
            #print(mlp.y_for_this_batch )


            y_unrolled = []
            for i in mlp.y_for_this_batch:
                y_unrolled.append(i[0])


        ###########------------First FOrward Pass and then backward pass---------
            y_unrolled = np.reshape(y_unrolled,(1,int(num_examples_per_batch)))
            #print(y_unrolled.shape)
            mlp.y_for_this_batch = np.transpose(y_unrolled)
            #time.sleep(11)







            #__________________________________________________________________TRAINING_____________________________________________________________________________
            loss, unwanted_acc, dE_by_dW2,dE_by_db2,dE_by_dW1,dE_by_db1 = mlp.train(mlp.x_for_this_batch,mlp.y_for_this_batch,int(learning_rate[lr]), inertia_of_momentum[iner], l2_penalty_factor[penalty])
            #print("Cross Entropy loss during training is ",loss)




            #___________________________________________________________________WEIGHT UPDATE_________________________________________________________________________________
            # print("bias 2 is ",mlp.linear_transform_object_second.bias)
            # time.sleep(2)
            mlp.weight_update(loss, dE_by_dW2, dE_by_db2, dE_by_dW1, dE_by_db1, learning_rate[lr], direction_for_W2, direction_for_W1, direction_for_b2, direction_for_b1, inertia_of_momentum[iner], l2_penalty_factor[penalty])


            total_loss_for_epoch = total_loss_for_epoch + loss


            #print('\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(epoch + 1,b + 1,total_loss_for_epoch,),end='', )

            #after each mini bach update you want to update the momentum value
            sys.stdout.flush()

        #print(" total_loss_for_epoch",epoch_num," is ",total_loss_for_epoch)

        #reset weights after each epoch
        # mlp.linear_transform_object_first = LinearTransform(0.001 * np.random.randn(input_dims, num_of_hidden_nodes),0.001 * np.full((num_of_hidden_nodes,int(num_examples_per_batch)),np.zeros((num_of_hidden_nodes,1))))#-----------------------MANUALLY done
        #
        #
        #
        #
        # mlp.linear_transform_object_second = LinearTransform(0.001 * np.random.randn(num_of_hidden_nodes, 1), 0.001 * np.full((1,int(num_examples_per_batch)),np.zeros(1)))

        # direction_for_W2 = np.zeros(mlp.linear_transform_object_second.weights.shape)
        # direction_for_W1 = np.zeros(mlp.linear_transform_object_first.weights.shape)
        # direction_for_b2 = np.zeros(mlp.linear_transform_object_second.bias.shape)
        # direction_for_b1 = np.zeros(mlp.linear_transform_object_first.bias.shape)
        # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy






        #------------------------------------Lets test the network------------------------------------------------------------------------------

        training_data_accuracy_for_this_epoch = mlp.evaluate(train_x, train_y,0)
        Training_accuracy_list.append(training_data_accuracy_for_this_epoch)


        testing_data_accuracy_for_this_epoch = mlp.evaluate(test_x, test_y,1)
        Testing_accuracy_list.append(testing_data_accuracy_for_this_epoch)



        #plot the results
        epoch_list_for_plot.append(epoch_num)
        #plt.plot([1,2,3],[3,4,5])

        print(" Epoch list Training Accuracy Testing Accuracy ", epoch_list_for_plot, Training_accuracy_list, Testing_accuracy_list)
        plt.plot(epoch_list_for_plot, Training_accuracy_list, 'g', label="Training Accuracy") #pass array or list
        plt.plot(epoch_list_for_plot, Testing_accuracy_list, 'r', label="Testing Accuracy")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Number of Accurate Predictions")
        plt.title("Number of Epochs VS Accuracies")


        epoch_num +=1
        #print('Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(training_loss_for_this_epoch,100. * train_accuracy,))
        # print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(testing_loss_for_this_epoch, 100. * test_accuracy,))
        #print("2nd weights W2 after the epoch no ", epoch_num, " is ",mlp.linear_transform_object_second.weights)
        #time.sleep(10)

    plt.show()

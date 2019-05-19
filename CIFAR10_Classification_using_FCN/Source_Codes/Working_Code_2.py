from __future__ import division
from __future__ import print_function

import sys
try:
   import _pickle as pickle
except:
   import pickle

import numpy as np
from scipy import special
import time


class LinearTransform(object):

    def __init__(self, weights, b):
        self.weights= weights * 0.01
        self.bias = b



    def forward(self, x):

        #Never do this because it does element-wise and this gives an error
        #linear_transformation_forward = x * self.weights# + self.bias

        #print("check",linear_transformation_forward)
        #time.sleep(5)
        linear_transformation_forward = np.dot(x, self.weights) + self.bias
        return linear_transformation_forward


    def forward_2(self,x):

        #print("shapes of x, self.weights, bias", self.weights.shape,x.shape,self.bias.shape)
        #time.sleep(1)
        batch_linear_summation_without_sigmoid = np.dot(np.transpose(self.weights),x)+self.bias
        #print("batch_linear_summation_without_sigmoid",batch_linear_summation_without_sigmoid)
        #time.sleep(222)
        return (batch_linear_summation_without_sigmoid)


    def backward(self, grad_output,input_to_calc_its_grad_op):

        #print("linear transform 2 ko backward bhitra ",np.transpose(grad_output).shape, np.transpose(self.weights).shape) #dE by dop dus ota hunuparxa for each example
        #print("here inside sha[pes ",np.transpose(grad_output).shape, np.transpose(self.weights).shape)


        dE_by_dZ1_to_return = np.dot(np.transpose(input_to_calc_its_grad_op), grad_output)
        return dE_by_dZ1_to_return


    # def weight_update(self, direction_for_old_weights, direction_for_old_bias, dE_by_dW, dE_by_db, learning_rate=0.0, inertial_of_momentum=0.0, l2_penalty_factor=0.0):
    #
    #     direction_for_new_weights = inertial_of_momentum * direction_for_old_weights - learning_rate * (dE_by_dW + l2_penalty_factor * self.weights)
    #     self.weights = self.weights + direction_for_new_weights
    #


    def weight_update(self, direction_for_old_weights, direction_for_old_bias, dE_by_dW, dE_by_db, learning_rate, inertial_of_momentum, l2_penalty_factor):

        #print("diretion chevk",direction_for_old_weights)



        direction_for_new_weights = inertial_of_momentum * direction_for_old_weights - learning_rate * (dE_by_dW + l2_penalty_factor * self.weights)
        self.weights = self.weights + direction_for_new_weights

        direction_for_new_bias = inertial_of_momentum * direction_for_old_bias - learning_rate * (dE_by_db + l2_penalty_factor * self.bias)
        self.bias = self.bias + direction_for_new_bias

        return direction_for_new_weights, direction_for_new_bias




        

class ReLU(object):

    def __init__(self):
        print()

    def forward(self, x):

        #print("Relu Input is ",x)

        #relu_output = np.maximum(0,x) #This is a slower method

        #print("relu_output is",relu_output)
        #time.sleep(11)
        relu_forward_prop = (x * (x > 0)) #faster method thatn the previous one
        return relu_forward_prop

    def backward(self,grad_output,A1):
        A1 = (A1 > 0) * 1 #here we very rarely encounter input = 0, so we can ignore the input = o condition, in case we want to implement that, we can do as shown in the commented line below
        #A1 = (A1 ==0) *np.random.range(0.00000000,0.99999999)#take a random number among infinite numbers in between 0 and 1 (exclude 0 and 1 from your range)
        dE1_by_dA1_to_retuen = np.multiply(grad_output, A1)
        return dE1_by_dA1_to_retuen



class SigmoidCrossEntropy(object):

    def __init__(self):
        print()

    def forward(self, x):
        #self.input_values = x
        #sigmoid =  (1.0 / (1 + np.exp(-x)))#-------------------use expit for faster calculation
        return special.expit(x)

    def backward(self, y, y_cap, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):

        #print("---------------true and predicted",y.shape,y_cap.shape)
        dE_by_dA2 = (y_cap - y) #------------------its predicted minus truth, not truth minus predicted
        return dE_by_dA2
	

class MLP(object):

    def __init__(self, input_dims, hidden_units):

        self.input_dims = input_dims
        self.hidden_units = hidden_units
        #-----------------instantiate the layers
        self.input_to_hidden_layer = LinearTransform(np.random.randn(input_dims, hidden_units), np.ones((1, hidden_units)))
        self.hidden_to_output_layer = LinearTransform(np.random.randn(hidden_units, 1), np.ones((1, 1)))

        self.relu_object = ReLU()
        self.sigmoid = SigmoidCrossEntropy()

        #--------------------set different metrics to zero
        self.training_loss = 0.0
        self.test_loss = 0.0
        self.training_misclassification_rate = 0
        self.test_misclassification_rate = 0


        #---------------------iitialize the summations, activations, predictions to zero
        self.Z1 = 0.0
        self.Z2 = 0.0
        self.A1 = 0.0
        self.predicted_y_for_training_data = 0
        self.predicted_y_for_test_Data = 0

        #---------------------direction needs to be zero at the beginning for all the weights and biases
        self.old_direction_for_W1 = 0
        self.old_direction_for_b1 = 0
        self.old_direction_for_W2 = 0
        self.old_direction_for_b2 = 0

    def train(self, x_batch, y_batch, learning_rate, momentum, l2_penalty,train_or_test):



        if train_or_test == 0:

    #-------------------------------------Forward Propagation
            linear_transformation = self.input_to_hidden_layer.forward(x_batch)

            relu_output = self.relu_object.forward(linear_transformation)

            linear_transformation_2 = self.hidden_to_output_layer.forward(relu_output)

            sigmoid_output = self.sigmoid.forward(linear_transformation_2)

            self.predicted_y_for_training_data = sigmoid_output


            #------------------------------clip the y_cap value such that it lies inside the interval (0,1)
            self.predicted_y_for_training_data = np.clip(self.predicted_y_for_training_data, 1e-12, 1 - (1e-12))


            self.Z1 = relu_output
            self.Z2 = linear_transformation_2
            self.A1 = linear_transformation


            #--------------------------------Backward Propagation
            dE_by_dA2 = self.sigmoid.backward(y_batch, sigmoid_output)

            dA2_by_dW2 = self.Z1

            dE_by_dW2 = self.hidden_to_output_layer.backward(dE_by_dA2, dA2_by_dW2)

            #print("yo k ho ta",dE_by_dW2.shape)

            dE_by_dB2 = dE_by_dA2

            dA2_by_dZ1 = self.hidden_to_output_layer.weights

            dE_by_dZ1 = np.dot(dE_by_dA2, np.transpose(dA2_by_dZ1))

            dE_by_dA1 = self.relu_object.backward(dE_by_dZ1, self.A1)

            dE_by_dW1 = self.input_to_hidden_layer.backward(dE_by_dA1, x_batch)

            dE_by_dB1 = dE_by_dA1

    #----------------------------------------Update the weights and biases

            self.old_direction_for_W1, self.old_direction_for_b1 =  self.input_to_hidden_layer.weight_update(self.old_direction_for_W1, self.old_direction_for_b1, dE_by_dW1, dE_by_dB1, learning_rate, momentum, l2_penalty)
            self.old_direction_for_W2, self.old_direction_for_b2 =  self.hidden_to_output_layer.weight_update(self.old_direction_for_W2, self.old_direction_for_b2, dE_by_dW2, dE_by_dB2, learning_rate, momentum, l2_penalty)


    #----------------------------------------Calculate Losses

            self.training_loss = self.cross_entropy_loss_calculation(y_batch, self.predicted_y_for_training_data, l2_penalty)

    #-----------------------------------------Calculate misclassifications
            self.training_misclassification_rate = self.calculate_misclassification_rate(y_batch, self.predicted_y_for_training_data)

            return self.training_loss, self.training_misclassification_rate

        # else:
        #
        #
        #     linear_transformation = self.input_to_hidden_layer.forward(x_batch)
        #
        #     relu_output = self.relu_object.forward(linear_transformation)
        #
        #     linear_transformation_2 = self.hidden_to_output_layer.forward(relu_output)
        #
        #     sigmoid_output = self.sigmoid.forward(linear_transformation_2)
        #
        #     self.predicted_y_for_training_data = sigmoid_output
        #
        #
        #     #------------------------------clip the y_cap value such that it lies inside the interval (0,1)
        #     self.predicted_y_for_training_data = np.clip(self.predicted_y_for_training_data, 1e-12, 1 - (1e-12))
        #
        #
        #     self.Z1 = relu_output
        #     self.Z2 = linear_transformation_2
        #     self.A1 = linear_transformation
        #
        #
        #     #--------------------------------Backward Propagation
        #     dE_by_dA2 = self.sigmoid.backward(y_batch, sigmoid_output)
        #
        #     dA2_by_dW2 = self.Z1
        #
        #     dE_by_dW2 = self.hidden_to_output_layer.backward(dE_by_dA2, dA2_by_dW2)
        #
        #     dE_by_dB2 = dE_by_dA2
        #
        #     dA2_by_dZ1 = self.hidden_to_output_layer.return_weights()
        #
        #     dE_by_dZ1 = np.dot(dE_by_dA2, np.transpose(dA2_by_dZ1))
        #
        #     dE_by_dA1 = self.relu_object.backward(dE_by_dZ1, self.A1)
        #
        #     dE_by_dW1 = self.input_to_hidden_layer.backward(dE_by_dA1, x_batch)
        #
        #     dE_by_dB1 = dE_by_dA1
        #
        #
        #     # print("\n\n\nFollowing are the dimensions of all the components")
        #     # print("x_for_this_batch shape",x_batch.shape)
        #     # print("\n\ny for this batch",y_batch.shape)
        #     # print("\nA1",mlp.A1)
        #     # print("\nZ1",mlp.Z1)
        #     # print("\nA2",mlp.A2)
        #     # print("\nZ2",mlp.y_cap)
        #     # # print("\ny cap transpose ",y_cap_transposed.shape)
        #     # # print("\ndE+by_dy_cap ",dE_by_dy_cap.shape)
        #     # print("\ndE_by_dA2 ",dE_by_dA2.shape)
        #     # print("\ndE_by_dW2 ",dE_by_dW2.shape)
        #     # #print("\nsecpn layer weights",mlp.linear_transform_object_second.weights.shape)
        #     # print("\ndE_by_db2 ",mlp.dE_by_db2.shape)
        #     # print("\ndE_by_dZ1 ",mlp.dE_by_dZ1.shape)
        #     # print("\ndE_by_dA1 ",mlp.dE_by_dA1.shape)
        #     # print("\ndE_by_dW1 ",dE_by_dW1.shape)
        #     # print("dE_by_db1 ",mlp.dE_by_db1.shape)
        #     #
        #     # print(" W1 is ",mlp.input_to_hidden_layer.weights.shape)
        #     # print(("W2 is ",mlp.hidden_to_output_layer.weights.shape))
        #     # time.sleep(10000)
        #     #
        #
        #
        #


    #----------------------------------------Update the weights and biases

            self.old_direction_for_W1, self.old_direction_for_b1 =  self.input_to_hidden_layer.weight_update(self.old_direction_for_W1, self.old_direction_for_b1, dE_by_dW1, dE_by_dB1, learning_rate, momentum, l2_penalty)
            self.old_direction_for_W2, self.old_direction_for_b2 =  self.hidden_to_output_layer.weight_update(self.old_direction_for_W2, self.old_direction_for_b2, dE_by_dW2, dE_by_dB2, learning_rate, momentum, l2_penalty)


    #----------------------------------------Calculate Losses

            self.training_loss = self.cross_entropy_loss_calculation(y_batch, self.predicted_y_for_training_data, l2_penalty)

    #-----------------------------------------Calculate misclassifications
            self.training_misclassification_rate = self.calculate_misclassification_rate(y_batch, self.predicted_y_for_training_data)

            return self.training_loss, self.training_misclassification_rate

    def calculate_metrics_for_CIFAR_test_data(self, x, y, l2_penalty,is_test_data):

        if is_test_data == 1:

            linear_transformation_ip_to_relu_layer = self.input_to_hidden_layer.forward(x)
            relu_obj_output = self.relu_object.forward(linear_transformation_ip_to_relu_layer)
            #print("relu obj",relu_obj_output)
            #time.sleep(3)
            linear_transformation_from_relu_to_output = self.hidden_to_output_layer.forward(relu_obj_output)
            y_cap_output_layer = self.sigmoid.forward(linear_transformation_from_relu_to_output)

            self.predicted_y_for_test_Data = y_cap_output_layer
            self.predicted_y_for_test_Data = np.clip(self.predicted_y_for_test_Data, 1e-12, 1 - (1e-12))

            self.test_loss = self.cross_entropy_loss_calculation(y, self.predicted_y_for_test_Data, l2_penalty)
            self.test_misclassification_rate = self.calculate_misclassification_rate(y, self.predicted_y_for_test_Data)

            return self.test_loss, self.test_misclassification_rate

        else:
            print()
            #---------------------------------remove this part later, you can handle the cases together
            # linear_transformation_ip_to_relu_layer = self.input_to_hidden_layer.forward(x)
            # relu_obj_output = self.relu_object.forward(linear_transformation_ip_to_relu_layer)
            # linear_transformation_from_relu_to_output = self.hidden_to_output_layer.forward(relu_obj_output)
            # y_cap_output_layer = self.sigmoid.forward(linear_transformation_from_relu_to_output)
            #
            # self.predicted_y_for_test_Data = y_cap_output_layer
            # self.predicted_y_for_test_Data = np.clip(self.predicted_y_for_test_Data, 1e-12, 1 - (1e-12))
            #
            # self.test_loss = self.cross_entropy_loss_calculation(y, self.predicted_y_for_test_Data, l2_penalty)
            # self.test_misclassification_rate = self.calculate_misclassification_rate(y, self.predicted_y_for_test_Data)
            #
            # return self.test_loss, self.test_misclassification_rate
            #
            #

    def cross_entropy_loss_calculation(self, y, y_cap, l2_penalty_factor):


         cumulative_cross_entropy = (np.sum(np.square(self.input_to_hidden_layer.weights)) + np.sum(np.square(self.hidden_to_output_layer.weights))) * (l2_penalty_factor/ (2 * len(y))) + (np.dot(np.transpose(y), np.log(y_cap + 1e-12)) + np.dot((1 - np.transpose(y)), np.log(1 - y_cap + 1e-12)))
         #print(cumulative)

         cross_entropy_loss_batch_average = (-1.0)* np.sum(cumulative_cross_entropy)/len(y)


         return cross_entropy_loss_batch_average

    def calculate_misclassification_rate(self, y, y_cap):

        y_cap = self.conversion_y_cap(y_cap)

        check = (y_cap == y)
        misclassifications = len((np.where(check==False))[0]) #shortcut
        #print("checkssss",misclassifications)
        #time.sleep(11
        return misclassifications

    def conversion_y_cap(self, y_cap):
        prediction_to_return = np.where(y_cap >= 0.5, np.ones(y_cap.shape), np.zeros(y_cap.shape))
        return prediction_to_return
    



def normalization(x):

    normalized_data_features = (x - np.amin(x, axis=0)) / np.amax(x, axis=0)- np.min(x, axis=0)
    return normalized_data_features



if __name__ == '__main__':
    if(sys.version_info[0] < 3):
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))

    else:
	    data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')

    train_x = data[b'train_data']
    train_y = data[b'train_labels']
    test_x = data[b'test_data']
    test_y = data[b'test_labels']

    #normalization
    train_x = normalization(train_x)
    test_x = normalization(test_x)

    num_examples, input_dims = train_x.shape
    total_training_examples = 10000
    total_testing_examples = 2000
    num_epochs = 1000
    num_batches = 100
    hidden_units = 100

    learning_rate = 0.0001

    print("learning rate is ",learning_rate)
    momentum = 0.7
    l2_penalty = 0.0001
    print("l2 penalty is ",l2_penalty)
    number_of_examples_per_batch = num_examples / num_batches
    print("no of hidden units is ",hidden_units, number_of_examples_per_batch)

    mlp = MLP(input_dims, hidden_units)
    print("momentum is: ",momentum)
    for epoch_num in range(num_epochs):
        print("\nEpoch Number: ", epoch_num)

        total_training_misclassification = 0
        total_testing_misclassification = 0
        cumulative_training_loss = 0
        cumulative_test_loss = 0


        #===================flag if its training data or testing data for mlp.train(function
        flag_for_test_data = 0

        for batch_to_use in range(num_batches):

            batch_starts_at_position = int((num_examples) / num_batches) * batch_to_use
            batch_ends_at_position = int((num_examples) / num_batches) * (batch_to_use + 1)

            #print("batch check ",batch_starts_at_position,batch_ends_at_position)
            #time.sleep(3)

            x_batch = train_x[batch_starts_at_position: batch_ends_at_position]
            y_batch = train_y[batch_starts_at_position: batch_ends_at_position]

            batch_loss_train, batch_error_train = mlp.train(x_batch, y_batch, learning_rate, momentum, l2_penalty,flag_for_test_data)

            cumulative_training_loss = cumulative_training_loss+ batch_loss_train
            total_training_misclassification = total_training_misclassification + batch_error_train

            #print("anfaflslks",cumulative_training_loss,total_training_misclassification)

        cumulative_training_loss = cumulative_training_loss / num_batches


        flag_for_test_data = 1

        for batch_to_use in range(int(total_testing_examples / number_of_examples_per_batch)):
            batch_starts_at_position = int((num_examples) / num_batches) * batch_to_use
            batch_ends_at_position = int((num_examples ) / num_batches) * (batch_to_use + 1)


            #print("batch check ",batch_starts_at_position,batch_ends_at_position)
            #time.sleep(3
            x_batch_test = test_x[batch_starts_at_position: batch_ends_at_position]
            y_batch_test = test_y[batch_starts_at_position: batch_ends_at_position]

            batch_loss_test, batch_error_test = mlp.calculate_metrics_for_CIFAR_test_data(x_batch_test, y_batch_test, l2_penalty,flag_for_test_data)

            cumulative_test_loss = cumulative_test_loss + batch_loss_test
            total_testing_misclassification = total_testing_misclassification + batch_error_test

        flag_for_test_data = 0 #-------------------------------reset flag




        cumulative_test_loss = cumulative_test_loss / num_batches
        train_accuracy = (total_training_examples - total_training_misclassification) / total_training_examples
        test_accuracy = (total_testing_examples - total_testing_misclassification) / total_testing_examples

        print('    Training Loss and Training Accuracy are: {:.3f} {:.2f}%'.format(cumulative_training_loss, 100.0 * train_accuracy,))
        print('    Testing Loss and Testing Accuracy are:  {:.3f} {:.2f}%'.format(cumulative_test_loss, 100.0 * test_accuracy,))


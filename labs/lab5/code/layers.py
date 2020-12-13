'''This file contains the implementations of the layers required by your neural network

For each layer you need to implement the forward and backward pass. You can add helper functions if you need, or have extra variables in the init function

Each layer is of the form - 
class Layer():
    def __init__(args):
        *Initializes stuff*

    def forward(self,X):
        # X is of shape n x (size), where (size) depends on layer
        
        # Do some computations
        # Store activations_current
        return X

    def backward(self, lr, activation_prev, delta):
        """
        # lr - learning rate
        # delta - del_error / del_activations_current
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        """
        # Compute gradients wrt trainable parameters
        # Update parameters
        # Compute gradient wrt input to this layer
        # Return del_error/del_activation_prev
'''
import numpy as np

class FullyConnectedLayer:
    def __init__(self, in_nodes, out_nodes, activation):
        # Method to initialize a Fully Connected Layer
        # Parameters
        # in_nodes - number of input nodes of this layer
        # out_nodes - number of output nodes of this layer
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.activation = activation   # string having values 'relu' or 'softmax', activation function to use
        # Stores the outgoing summation of weights * feautres 
        self.data = None

        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))    
        self.biases = np.random.normal(0,0.1, (1, out_nodes))
        ###############################################
        # NOTE: You must NOT change the above code but you can add extra variables if necessary 

    def forwardpass(self, X):
        '''
                
        Arguments:
            X  -- activation matrix       :[n X self.in_nodes]
        Return:
            activation matrix      :[n X self.out_nodes]
        '''
        # TODO
        if self.activation == 'relu':
            self.data = relu_of_X((X@self.weights)+self.biases)
            return self.data
        elif self.activation == 'softmax':
            self.data =  softmax_of_X((X@self.weights)+self.biases)
            return self.data

        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        # END TODO      
    def backwardpass(self, lr, activation_prev, delta):
        '''
        # lr - learning rate
        # delta - del_error / del_activations_current  : 
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        '''

        # TODO 
        if self.activation == 'relu':
            memoised_delta = gradient_relu_of_X(self.data, delta)
        elif self.activation == 'softmax':
            memoised_delta = gradient_softmax_of_X(self.data, delta)
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        # n, ic | n, oc
        self.grad_w = (activation_prev.T @ memoised_delta)/self.data.shape[0]
        #n , oc
        self.grad_b = memoised_delta.sum(0)/self.data.shape[0]

        # n, oc | ic, oc
        new_delta = memoised_delta@self.weights.T
        
        self.weights -= lr*self.grad_w
        self.biases -= lr*self.grad_b

        # Compute gradient wrt input to this layer
        # Return del_error/del_activation_prev
        return new_delta

        # END TODO
class ConvolutionLayer:
    def __init__(self, in_channels, filter_size, numfilters, stride, activation):
        # Method to initialize a Convolution Layer
        # Parameters
        # in_channels - list of 3 elements denoting size of input for convolution layer
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer
        # numfilters  - number of feature maps (denoting output depth)
        # stride      - stride to used during convolution forward pass
        # activation  - can be relu or None
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride
        self.activation = activation
        self.out_depth = numfilters
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

        # Stores the outgoing summation of weights * feautres 
        self.data = None
        
        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))   
        self.biases = np.random.normal(0,0.1,self.out_depth)
        

    def forwardpass(self, X):
        # INPUT activation matrix       :[n X self.in_depth X self.in_row X self.in_col]
        # OUTPUT activation matrix      :[n X self.out_depth X self.out_row X self.out_col]

        # TODO
        self.data = np.zeros((X.shape[0],self.out_depth,self.out_row, self.out_col))
        # och, ich, fr*fc
        reshaped_weights = self.weights.reshape(self.out_depth, self.in_depth, -1)
        if self.activation == 'relu':
            # n, och, or, oc
            self.data = np.zeros((X.shape[0],self.out_depth,self.out_row, self.out_col))
            # n, och, ich, or, oc, fr*fc
            self.linkages = np.zeros((X.shape[0],self.out_depth, X.shape[1],self.out_row, self.out_col, self.filter_row*self.filter_col))
            for ri in range(self.out_row):
                for ci in range(self.out_col):
                    offset_r = ri*self.stride
                    offset_c = ci*self.stride

                    # n,ich,fr*fc
                    patch = X[:,:,offset_r:offset_r+self.filter_row,offset_c:offset_c+self.filter_col].reshape(X.shape[0], X.shape[1],-1)
                    # [n,1,ich,fr*fc | 1, och, ich,fr*fc -> n, och,ich, fr*fc] 
                    # n, och, ich, fr*fc1, -> n,och | 1, och
                    weighted_patch = ((np.expand_dims(patch,1)*np.expand_dims(reshaped_weights,0)).sum(3).sum(2) + np.expand_dims(self.biases,0))
                    self.data[:,:, ri, ci] = weighted_patch
                    # n, och, ich, fr*fc
                    self.linkages[:,:,:,ri, ci,:] = np.expand_dims(reshaped_weights, 0)
            return relu_of_X(self.data)
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        
        ###############################################
        # END TODO
    def backwardpass(self, lr, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # Update self.weights and self.biases for this layer by backpropagation
        # TODO

        ###############################################
        if self.activation == 'relu':
        	# n, och, or, oc
            memoised_delta = gradient_relu_of_X(self.data, delta)

            # n, ich, ir, ic
            new_delta = np.zeros(activation_prev.shape)
            # och,
            grad_b = memoised_delta.sum(2).sum(2).mean(0)
            # och, ich, fr, fc
            grad_w = np.zeros(self.weights.shape)
            for ri in range(self.out_row):
                for ci in range(self.out_col):
                    offset_r = ri*self.stride
                    offset_c = ci*self.stride
                    
                    # n, och, ich, fr*fc | n, och, 1 -> n, ich, fr*fc 
                    patch = (self.linkages[:,:,:,ri,ci,:]*np.expand_dims(np.expand_dims(memoised_delta[:,:,ri,ci], 2), 3)).sum(1)
                    # n, ich, fr,fc | n,ich, fr,fc
                    new_delta[:,:,offset_r:offset_r+self.filter_row,offset_c:offset_c+self.filter_col] += patch.reshape(patch.shape[0], patch.shape[1], self.filter_row,self.filter_col)

                    # n,och, 1, 1, 1
                    memoised_delta_chunk = np.expand_dims(np.expand_dims(np.expand_dims(memoised_delta[:,:,ri,ci],2),3),4)
                    # n, 1, ich, fr, fc
                    activation_prev_chunk = np.expand_dims(activation_prev[:,:,offset_r:offset_r+self.filter_row,offset_c:offset_c+self.filter_col], 1)
                    # n, och, ich, fr, fc 
                    grad_w += (memoised_delta_chunk*activation_prev_chunk).mean(0)


            self.weights -= lr*grad_w
            self.biases -= lr*grad_b
            return new_delta
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        ###############################################

        # END TODO
    
class AvgPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        # TODO
        # n, ich, ro, co
        self.data = np.zeros((X.shape[0],X.shape[1],self.out_row, self.out_col))
        # n, ich, ro, co, fr*fc
        self.linkages = np.zeros((X.shape[0],X.shape[1],self.out_row,self.out_col,self.filter_row*self.filter_col))
        for ri in range(self.out_row):
            for ci in range(self.out_col):
                offset_r = ri*self.stride
                offset_c = ci*self.stride
                
                # n, ich, fr*fc
                patch = X[:,:,offset_r:offset_r+self.filter_row,offset_c:offset_c+self.filter_col].reshape(X.shape[0], X.shape[1],-1)
                # n, ich
                self.data[:,:, ri, ci] = patch.mean(2)
                # n, ich, fr*fc
                self.linkages[:,:, ri, ci,:] = 1/(self.filter_row*self.filter_col)
        return self.data
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # TODO
        ans = np.zeros(activation_prev.shape)
        for ri in range(self.out_row):
            for ci in range(self.out_col):
                offset_r = ri*self.stride
                offset_c = ci*self.stride
                # n, ich, fr*fc | n,och,1
                patch = self.linkages[:,:,ri,ci,:]*np.expand_dims(delta[:,:,ri,ci], 2)
                # n, ich, fr,fc
                ans[:,:,offset_r:offset_r+self.filter_row,offset_c:offset_c+self.filter_col] += patch.reshape(ans.shape[0], ans.shape[1], self.filter_row, self.filter_col)
        return ans
        # END TODO
        ###############################################



class MaxPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        # TODO
        # n, ich, or, oc
        self.data = np.zeros((X.shape[0],X.shape[1],self.out_row, self.out_col))
        # n, ich, or, oc, fr*fc
        self.linkages = np.zeros((X.shape[0],X.shape[1],self.out_row,self.out_col,self.filter_row*self.filter_col))
        for ri in range(self.out_row):
            for ci in range(self.out_col):
                offset_r = ri*self.stride
                offset_c = ci*self.stride
                # n, ich, fr*fc
                patch = X[:,:,offset_r:offset_r+self.filter_row,offset_c:offset_c+self.filter_col].reshape(X.shape[0], X.shape[1],-1)
                # n, ich
                self.data[:,:, ri, ci] = patch.max(2)
                # n*ich,fr*fc
                temp_patch = patch.reshape(-1,self.filter_row*self.filter_col)
                # n*ich,fr*fc
                temp = self.linkages[:,:, ri, ci,:].reshape(-1,self.filter_row*self.filter_col)
                # n*ich,fr*fc
                temp[np.arange(X.shape[0]*X.shape[1]),np.argmax(temp_patch,1)] = 1
                self.linkages[:,:,ri,ci,:] = temp.reshape(X.shape[0],X.shape[1],-1)
        return self.data
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        # TODO
        ans = np.zeros(activation_prev.shape)
        for ri in range(self.out_row):
            for ci in range(self.out_col):
                offset_r = ri*self.stride
                offset_c = ci*self.stride
                # n, ich, fr*fc | n, och, 1
                patch = self.linkages[:,:,ri,ci,:]*np.expand_dims(delta[:,:,ri,ci], 2)
                # n, ich, fr,fc
                ans[:,:,offset_r:offset_r+self.filter_row,offset_c:offset_c+self.filter_col] += patch.reshape(ans.shape[0], ans.shape[1], self.filter_row, self.filter_col)
        return ans
        # END TODO
        ###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        # TODO
       	self.n, self.ich, self.ir, self.ic = X.shape 
        return X.reshape(self.n, -1)
    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.n, self.ich, self.ir, self.ic)
        # END TODO

# Function for the activation and its derivative
def relu_of_X(X):

    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu
    # TODO
    ans = X.copy()
    indices = ans <= 0
    ans[indices] = 0
    return ans
    # END TODO 
    
def gradient_relu_of_X(X, delta):
    # Input
    # Note that these shapes are specified for FullyConnectedLayers, the function also needs to work with ConvolutionalLayer
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu amd during backwardpass
    
    # TODO
    ans = np.ones(X.shape)
    ans[X<=0] = 0
    return ans*delta
    # END TODO

def softmax_of_X(X):
    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax
    
    # TODO
    ans = np.exp(X)
    return ans/(np.sum(ans,1,keepdims=True))
    # END TODO  
def gradient_softmax_of_X(X, delta):
    # Input
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax amd during backwardpass
    # Hint: You might need to compute Jacobian first

    # TODO
    n, c = X.shape
    Jacobs = np.zeros((n,c,c))
    diag = np.arange(c)
    Jacobs[:,diag,diag] = X
    x1 = np.expand_dims(X,2)
    x2 = np.expand_dims(X,1)
    Jacobs -= x1*x2

    return (Jacobs*np.expand_dims(delta,2)).sum(1)
    # END TODO

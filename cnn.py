from builtins import object
import numpy as np

from layers import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - linear - relu - linear - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # Initialize Weights and Biases
        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        # Assuming a shape identical to the input image for the conv layer output
        self.params['W2'] = weight_scale * np.random.randn(num_filters * H * W // 4, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in layers.py                  #
        ############################################################################
#        conv - relu - 2x2 max pool - linear - relu - linear - softmax
        conv_out, conv_cache = conv_forward(X, W1, b1, conv_param)
        relu_out, relu_cache = relu_forward(conv_out)
        max_out, max_cache = max_pool_forward(relu_out, pool_param)
        linear_out, linear_cache = linear_forward(max_out, W2, b2)
        relu_out, relu2_cache = relu_forward(linear_out)
        scores, linear2_cache = linear_forward(relu_out, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = None, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dScores = softmax_loss(scores, y)
        for key in self.params.keys():
            if "W" in key:
                loss += 0.5 * self.reg * np.sum(np.square(self.params[key])) 
                
        dLinear2_x, dLinear2_w, dLinear2_b = linear_backward(dScores, linear2_cache)
        dRelu2 = relu_backward(dLinear2_x, relu2_cache)
        dLinear_x, dLinear_w, dLinear_b = linear_backward(dRelu2, linear_cache)
        dMax = max_pool_backward(dLinear_x, max_cache)
        dRelu = relu_backward(dMax, relu_cache)
        dConv_x, dConv_w, dConv_b = conv_backward(dRelu, conv_cache)
        
        grads["W1"] = dConv_w + 0.5 * self.reg * 2 * W1
        grads["b1"] = dConv_b + 0.5 * self.reg * 2 * b1
        grads["W2"] = dLinear_w + 0.5 * self.reg * 2 * W2
        grads["b2"] = dLinear_b + 0.5 * self.reg * 2 * b2
        grads["W3"] = dLinear2_w + 0.5 * self.reg * 2 * W3
        grads["b3"] = dLinear2_b + 0.5 * self.reg * 2 * b3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def train(self, X, y, learning_rate=1e-3, num_epochs=10,
              batch_size=2, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
        training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
        means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.
        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train = X.shape[0]
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        loss_history = []
        acc_history = []

        for epoch in range(1, num_epochs):
            loss_epoch = []
            acc_epoch = []

            for it in range(iterations_per_epoch):
                X_batch = None
                y_batch = None

                indices = np.random.choice(num_train, batch_size)
                X_batch = X[indices]
                y_batch = y[indices]

                # evaluate loss and gradient
                loss, grads = self.loss(X_batch, y_batch)
                loss_epoch.append(loss)

                # perform parameter update
                #########################################################################
                # TODO:                                                                 #
                # Update the weights using the gradient and the learning rate.          #
                #########################################################################
                for key in self.params.keys():
                    self.params[key] -= learning_rate * grads[key]
                #########################################################################
                #                       END OF YOUR CODE                                #
                #########################################################################

                acc_epoch.append(np.mean(self.predict(X_batch) == y_batch))

            loss_history.append(np.mean(loss_epoch))
            acc_history.append(np.mean(acc_epoch))

            if verbose and epoch % 10 == 0:
                print('epoch {} / {} : loss {}'.format(epoch, num_epochs, loss), end='\r')

        if verbose:
            print(''.ljust(70), end='\r')

        return loss_history, acc_history

    def predict(self, X):
        y_pred = np.zeros(X.shape[1])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        conv_out, _ = conv_forward(X, W1, b1, conv_param)
        relu_out, _ = relu_forward(conv_out)
        max_out, _ = max_pool_forward(relu_out, pool_param)
        linear_out, _ = linear_forward(max_out, W2, b2)
        relu_out, _ = relu_forward(linear_out)
        linear2_out, _ = linear_forward(relu_out, W3, b3)
        y_pred, _ = softmax_forward(linear2_out)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

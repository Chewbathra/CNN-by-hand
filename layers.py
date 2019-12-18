from builtins import range
import numpy as np


def linear_forward(x, w, b):
    """
    Computes the forward pass for an linear (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)
    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the linear forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    x_reshaped = np.reshape(x, (x.shape[0], np.prod(x[0].shape)))
    out = np.matmul(x_reshaped, w) + b    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def linear_backward(dout, cache):
    """
    Computes the backward pass for an linear layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    x_reshaped = np.reshape(x, (x.shape[0], np.prod(x[0].shape)))
    dw = x_reshaped.T.dot(dout)
    dx = w.dot(dout.T).T.reshape(x.shape)
    db = np.sum(dout, axis=0)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None

    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(x, 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = np.where(x > 0, 1, 0) * dout
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def conv_forward(x, w, b, conv_param):
    """
    An implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    padding = conv_param['pad']
    stride = conv_param['stride']
    N = x.shape[0]
    H = x.shape[2]
    W = x.shape[3]
    F = w.shape[0]
    HH = w.shape[2]
    WW = w.shape[3]
    
    h_prim = int(1 + (H + 2 * padding - HH) / stride)
    w_prim = int(1 + (W + 2 * padding - WW) / stride)
    out = np.zeros((N, F, h_prim, w_prim))
    
    #Naive version
#    for n in range(N): #Iterate trough each x
#        current_H =  int(H + padding * 2) #Get height of current_x with padding
#        current_W = int(W + padding * 2) #Get width of current_x with padding
#        current_x = np.zeros((C, current_H, current_W)) #Set current_x
#            
#        for f in range(F): #Iterate trough filters
#            for c in range(C): #Iterate trough channels
#                current_x[c] = np.pad(x[n,c], padding, mode="constant", constant_values=(0)) #Padding
#                i_out = 0 #H index in the out array
#                i = 0
#                i_range = i + HH # End H index of the filter on x H axis
#                while i_range <= current_H: #Move the filter on H axis
#                    j_out = 0
#                    j = 0 #W index in the out array
#                    j_range = j + WW #End W index of the filter on x W axis
#                    while j_range <= current_W: #Move the filter on W axis
#                        out[n,f,i_out,j_out] += np.sum(current_x[c, i:i_range, j:j_range] * w[f,c])
#                        j_out += 1
#                        j += stride
#                        j_range = j + WW
#                    i_out += 1
#                    i += stride
#                    i_range = i + HH
#            
#            out [n,f] = out[n,f] + b[f]
    
    # Less naive version
#    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=(0))
#    for n in range(N): #Iterate trough each x      
#        for f in range(F): #Iterate trough filters
#            i = 0
#            for j in range(h_prim): #Move the filter on H axis
#                j = 0 #W index in the out array
#                for i in range(w_prim): #Move the filter on W axis
#                    out[n, f, i, j] = np.sum(x_padded[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] * w[f]) + b[f]
    
    # Version with multiple dimensions matrix sum
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=(0))
    for n in range(N): #Iterate trough each x      
        for j in range(h_prim): #Iterate on H axis
            j_stride = j * stride #Select the current y index in x
            for i in range(w_prim): #Iterate on W axis
                i_stride = i * stride #Select the current x index in x
                # Multiply the filter with the selected values on x
                out[n, :, j, i] = np.sum(x_padded[n, :, j_stride:j_stride + HH, i_stride:i_stride + WW] * w, axis=(1,2,3)) + b
        
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward(dout, cache):
    """
    An implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    padding = conv_param['pad']
    stride = conv_param['stride']
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    F = w.shape[0]
    HH = w.shape[2]
    WW = w.shape[2]

    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=(0))
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)  
    db = np.zeros(b.shape)
    
    for n in range(N):
        for i in range(H):
            for j in range(W):
                for f in range(F):
                    for k in range(dout.shape[2]):
                        for l in range(dout.shape[3]):
                            mask1 = np.zeros_like(w[f, :, :, :])
                            mask2 = np.zeros_like(w[f, :, :, :])
                            if (i + padding - k * stride) < HH and (i + padding - k * stride) >= 0:
                                mask1[:, i + padding - k * stride, :] = 1.0
                            if (j + padding - l * stride) < WW and (j + padding - l * stride) >= 0:
                                mask2[:, :, j + padding - l * stride] = 1.0
                            w_masked = np.sum(w[f, :, :, :] * mask1 * mask2, axis=(1, 2))
                            dx[n, :, i, j] += dout[n, f, k, l] * w_masked
        
    
    for n in range(N):
        for f in range(F):
            db[f] = np.sum(dout[:, f, :, :]) #Set biais gradien t
            for j in range(HH):
                j_stride = j * stride
                for i in range(WW):
                    i_stride = i * stride
                    dw[f, :, j, i] += np.sum(x_padded[n, :, j_stride:j_stride+H, i_stride:i_stride+W] * dout[n, f, :, :], axis=(1,2))


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward(x, pool_param):
    """
    An implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    h_prim = int(1 + (H - pool_height) / stride)
    w_prim = int(1 + (W - pool_width) / stride)
    
    out = np.zeros((N, C, h_prim, w_prim))
    for j in range(h_prim):
        j_stride = j * stride
        for i in range(w_prim):
            i_stride = i * stride
            out[:, :, j, i] = np.max(x[:, :, j_stride:j_stride+pool_height, i_stride:i_stride+pool_width], axis=(2,3))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    An implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    h_prim = int(1 + (H - pool_height) / stride)
    w_prim = int(1 + (W - pool_width) / stride)
    
    dx = np.zeros(x.shape)

    for n in range(N):
        for c in range(C):
            for j in range(h_prim):
                j_stride = j * stride
                for i in range(w_prim):
                    i_stride = i * stride
                    current = x[n, c, j_stride:j_stride+pool_height, i_stride:i_stride+pool_width]
                    a,b = np.unravel_index(current.argmax(), current.shape)
                    dx[n, c, a + j_stride, b + i_stride] = dout[n,c,j,i]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def softmax_forward(x):
    out = None
    
#    x_norm = x- np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    out = exp / np.sum(exp)

    cache = x
    return out, cache


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


def linear_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = linear_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def linear_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = linear_backward(da, fc_cache)
    return dx, dw, db

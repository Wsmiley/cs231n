from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1] #10
    num_train = X.shape[0]
    for i in range(num_train):
      scores = X[i].dot(W) #算出xiW的每个类得分  (3072)*(3072,10)
      adjust_score = scores - np.max(scores)  
      loss += -np.log(np.exp(adjust_score[y[i]] / np.sum(np.exp(adjust_score))))
      for j in range(num_classes):
        prob = np.exp(adjust_score[j]) / np.sum(np.exp(adjust_score))
        if j==y[i]:
          dW[:,j] += (-1 + prob) * X[i]
        else:
          dW[:,j] += prob * X[i]

    loss = loss / num_train
    dW = dW / num_train
    loss += reg * np.sum(W * W)
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    x_source = np.dot(X,W) #
    adjust_scores = np.exp(x_source - np.max(x_source, axis=1).reshape(-1, 1))
    sum_scores = np.sum(adjust_scores, axis=1).reshape(-1, 1)
    class_prob = adjust_scores / sum_scores  # shape [N, C]
    #
    prob = class_prob[np.arange(num_train), y]
    #
    total_loss = -np.log(prob)
    loss = np.sum(total_loss) / num_train + reg * np.sum(W * W)
    #
    class_prob[np.arange(num_train), y] -= 1
    #
    dW = (X.T).dot(class_prob)
    dW = dW / num_train + 2 * reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

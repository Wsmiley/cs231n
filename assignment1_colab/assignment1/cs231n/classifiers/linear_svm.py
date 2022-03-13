from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1] #10
    num_train = X.shape[0]  #3073
    loss = 0.0
    for i in range(num_train):
      scores = X[i].dot(W) #算出xiW的每个类得分  (3072)*(3072,10)

      correct_class_score = scores[y[i]] #正确的分数
      for j in range(num_classes): 
        if j == y[i]:    #是这个分类就跳过，不是就计算margin
          continue
        margin = scores[j] - correct_class_score + 1  # note delta = 1 #公式
        if margin > 0:  #等于0就是分类正确
          loss += margin   #累加和
          dW[:,y[i]]=-X[i,:].T 
          dW[:,j]+=X[i,:].T  

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W) #total_loss = avg_loss + lambda *  sum(W*W)
    dW += reg * W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    x_source = np.dot(X,W) #
    x_source_correct = x_source[np.arange(num_train),y]
    x_source_correct = np.reshape(x_source_correct,(num_train,-1))
    margins = np.maximum(x_source-x_source_correct+1,0) #shape[N,C]
    margins[np.arange(num_train),y]=0
    loss = np.sum(margins)/num_train+reg*np.sum(W*W) #1/N*sum(margins)+regularization
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    margins[margins > 0] = 1
    #因为j=y[i]的那一个元素的grad要计算 >0 的那些次数次
    row_sum = np.sum(margins,axis=1)
    margins[np.arange(num_train),y] = -row_sum.T
    #把公式1和2合到一起计算了
    dW = np.dot(X.T,margins)
    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

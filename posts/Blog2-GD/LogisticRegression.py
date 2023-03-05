# Logistic Regression Class

import numpy as np 

class LogisticRegression:
    
    def __init__(self):
        self.w = None # Weight vector
        self.loss_history = None # Array of past losses
        self.score_history = None # Score history for accuracy



    """
    Logistic Regression Fit Algorithm
    Parameters: X (matrix), y (labels)
    Return: no return value

    In this algorithm, we fit a logistic regression model on our dataset.
    To do so, we perform the following operations

    1. Pad X matrix with 1's column (in order to perform dot products)
    2. Initialize w to random values
    3. While not done
        1. Update w using the gradient formula
        2. Calculate new loss using the empirical risk formula (explained below)
        3. Append new loss to loss_history
        4. Use np.isclose() to see if convergence has been reached
        5. If not, increment steps and set old loss = new loss

    """

    def fit(self, X, y, alpha=.1, max_epochs=1000):

        # Get shape of data
        n,p = np.shape(X)

        # Initialize values
        w = np.random.rand(p+1)
        X_ = np.append(X, np.ones((n, 1)), 1)

        # Main loop

        done = False
        prev_loss = np.inf
        history = []
        steps = 0

        
        while not done and steps < max_epochs:

            # Calculate new w
            w_ = w - (alpha * self.gradient(w, X_, y))
            w = w_
            self.w = w  

            # Get log loss
            new_loss = self.empirical_risk(X_, y, w)

            # Append new loss to history
            history.append(new_loss)

            # Check termination condition
            if np.isclose(new_loss, prev_loss) or steps >= max_epochs:          
                done = True
            else:
                prev_loss = new_loss
                steps += 1

        self.loss_history = history


    """
    Logistic Regression Stochastic Fit Algorithm
    
    Parameters: X (matrix), y (labels)
    Return: no return value

    In this algorithm, we fit a logistic regression model on our dataset.
    To do so, we perform the following operations

    1. Pad X matrix with 1's column (in order to perform dot products)
    2. Initialize w to random values
    3. Loop through max_epochs and do the following:
        1. Get an order of our data using the np.arange function
        2. Shuffle the matrix using the shuffle operation
        3. Pick the first k random points, compute the stochastic gradient, and then perform an update.
        4. Pick the next random points and repeat..
        5. When we have gone through all points, reshuffle them all randomly and proceed again.

    """


    def fit_stochastic(self, X, y, batch_size = 10, alpha=.1, max_epochs=1000):

       # Get shape of data
        n,p = np.shape(X)

        # Initialize values
        w = np.random.rand(p+1)
        X_ = np.append(X, np.ones((n, 1)), 1)

        # Main loop

        history = []

        for j in range(max_epochs): 

            order = np.arange(n)
            np.random.shuffle(order)

            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X_[batch,:]
                y_batch = y[batch]
                
                w_ = w - (alpha * self.gradient(w, x_batch, y_batch))
                w = w_
                self.w = w 

            # Get log loss
            new_loss = self.empirical_risk(X_, y, w)
                

            # Append new loss to history
            history.append(new_loss)

        self.loss_history = history
                

    """
    Sigmoid

    Parameters: z (np array)
    Returns: Result of sigmoid equation (np array)
    """

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    
    """
    Logistic Loss

    Parameters: y_hat (predicted values), y (real values)
    Returns: Result of Log Loss equation
    """
    
    def logistic_loss(self, y_hat, y): 
        return -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))
    

    """
    Gradient:

    Parameters: w (weight vector), X_ (padded matrix), y (real values)
    Returns: average of summed gradients
    """
    
    def gradient(self, w, X_, y):
        n = len(y)
        total = 0
        for i in range(n):
            total += (self.sigmoid(np.dot(X_[i], w)) - y[i]) * X_[i]
        return total/n
    
    """
    Predict

    Parameters: X (matrix)
    Returns: y_hat (predicted values)
    """
    
    def predict(self, X):   
        n,p = np.shape(X)
        X_ = np.append(X, np.ones((n, 1)), 1)
        return np.where(self.sigmoid(np.dot(X_, self.w)) > 0.5, 1, 0)
    

    """
    Empirical Risk

    Parameters: X (padded matrix), y (real values), w (weight vector)
    Returns: Loss (using log loss)
    """
    
    def empirical_risk(self, X_, y, w):
        y_hat = np.dot(X_, w)
        loss = np.mean(self.logistic_loss(y_hat, y))
        return loss
    
    """
    Score

    Parameters: X (matrix), y (real values)
    Returns: Accuracy of model on data
    """
    
    def score(self, X, y):
        y_hat = self.predict(X)
        return np.mean(y == y_hat)





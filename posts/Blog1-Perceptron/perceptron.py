
import numpy as np
np.random.seed()

class Perceptron:
    

    def __init__(self):
        self.w = None
        self.history = None
        self.accuracy = None

    

    """
    Perceptron algorithm
    X: nxp matrix, where n is observations, p is features
    y: Binary vector of size n
    """
    
    def fit(self, X, y, max_steps):
        
        # Change 0's to -1's in y vector

        y = [-1 if x == 0 else x for x in y]

        # Get shape of X
        n,p = np.shape(X)

        # Initialize function restraints
        loss = 1
        steps = 0

        # Initialize w, X_, history
        w_ = np.random.rand(p+1)
        X_ = np.append(X, np.ones((n, 1)), 1)
        history = []

        while loss != 0 and steps < max_steps:
            
            # First, choose random x
            rand_index = np.random.randint(0, n-1)
            
            # Set random observation
            xi = X_[rand_index]

            # Make prediction y and get real y
            y_ = np.sign(np.dot(xi, w_))
            yi = y[rand_index]

            # Check if correct
            # If not correct, update
            if y_ != yi:
                
                w1_ = w_ + yi*xi

                #Calculate Loss
                accuracy = self.get_accuracy(X_, y, w1_)
                loss = 1 - accuracy
                self.accuracy = accuracy

                #Update History
                history.append(accuracy)
                self.history = history
                w_ = w1_


                

            # Update steps
            steps += 1

            # Update w
            self.w = w_

    def get_accuracy(self, X_, y, w_):

        return ( (np.dot(X_, w_) ) * y > 0 ).mean()

    def score(self, X, y):
        return self.accuracy

    def predict(self, X):
        
        n,p = np.shape(X)
        X_ = np.append(X, np.ones((n, 1)), 1)
        w = self.w
        return np.sign(np.dot(X_, w)) 
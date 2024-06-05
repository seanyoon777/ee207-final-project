# -------------------------------------------------------------------------------------------------
# KalmanDecoder.py 
# -------------------------------------------------------------------------------------------------
# Baseline Kalman decoder for predicting fingertip or cursor velocity. 
# Model is optimized for energy efficiency, with original code from KordingLab's implementation.  
# More details in : https://arxiv.org/abs/1708.00909
# Sean Yoon [sean777@stanford.edu], Nika Zahedi [nzahedi@stanford.edu]
# -------------------------------------------------------------------------------------------------
# Created     : 2024-06-04
# Last update : 2024-06-04
# -------------------------------------------------------------------------------------------------
from Globals import *
# -------------------------------------------------------------------------------------------------

class KalmanDecoder: 
    def __init__(self, C=1, pred_dims=2): 
        """
        Initializes the Kalman filter decoder class. 

        Args:
            C : noise matrix, defaults to 1. 
        """
        self.C = C 
        self.pred_dims = 2
        self.model = None 
    
    @staticmethod
    @nb.jit(nopython=True)
    def update(X, Z, C): 
        """_summary_
        Computes time update parameters for discrete Kalman Filter. Follows nomenclature from Neural 
        Decoding of Cursor Motion using a Kalman Filter (Wu et al., 2003). 

        Args:
            X (_type_): _description_
            Z (_type_): _description_
            C (_type_): _description_
        """
        # converts to contiguous arrays and jit-compatible formats for energy efficiency
        X = np.ascontiguousarray(X, dtype='float32')
        Z = np.ascontiguousarray(Z, dtype='float32')
        
        # calculate the transition matrix from X_t to X_{t+1}
        X1 = X[:, :-1]
        X2 = X[:, 1:]
        A = np.linalg.lstsq(X1 @ X1.T, X2 @ X1.T, rcond=None)[0].T 
        
        # calculate covariance of transition matrix
        residual = X2 - A @ X1
        W = residual @ residual.T / X1.shape[1] / C

        # calculates the measurement matrix from X_t to X_{t+1}
        # This linearly relates hand state to neural firing
        H = np.linalg.lstsq(X @ X.T, Z @ X.T, rcond=None)[0].T
        
        # calculate the covariance of the measurement matrix
        residual = X - H @ X
        Q = residual @ residual.T / X1.shape[1]

        return A, W, H, Q 
    
    
    def fit(self, X_train, y_train): 
        """
        Trains the KF Decoder object. 

        Args:
            X_train : [samples x channels] shape array of neural data. 
            y_train : [samples x velocity_dims] shape array of outputs to predict. 

        Returns:
            _type_: _description_
        """
        # transpose and make data contiguous 
        X = np.ascontiguousarray(y_train.T, dtype='float32')
        Z = np.ascontiguousarray(X_train.T, dtype='float32')
        
        # compute matrices
        A, W, H, Q = self.update(X, Z, self.C)
        self.model = [A, W, H, Q] 


    def predict(self, X_test): 
        A, W, H, Q = self.model 
        y_test = self._predict(X_test, self.pred_dims, A, W, H, Q)
        return y_test 
    

    @staticmethod
    @nb.jit(nopython=True)
    def _predict(X_test, n_states, A, W, H, Q): 
        # transpose and make data contiguous 
        Z = np.ascontiguousarray(X_test.T, dtype='float32')
        
        # initialize 
        n_samples = Z.shape[1]
        states = np.zeros((n_states, n_samples), dtype='float32')  # state predictions
        P_m    = np.zeros((n_states, n_states), dtype='float32')   # prior error covariance
        P      = np.zeros((n_states, n_states), dtype='float32')   # posterior error covariance
        state  = np.zeros((n_states, 1), dtype='float32')          # initial state as zeros
        
        # make them contiguous 
        states = np.ascontiguousarray(states)
        P_m    = np.ascontiguousarray(P_m)
        P      = np.ascontiguousarray(P) 
        state  = np.ascontiguousarray(state)
        
        # kalman filter loop 
        for t in range(1, n_samples): 
            # predict state and covariance 
            state_m = A @ state 
            P_m = A @ P @ A.T + W 
            
            # calculate system uncertainty and kalman gain
            S = H @ P_m @ H.T + Q 
            K = P_m @ H.T @ np.linalg.inv(S)

            # update and store the updated state
            state = state_m + K @ (Z[:, t:t+1] - H @ state_m)
            P     = (np.eye(n_states) - K @ H) @ P_m 
            
            states[:, t] = state[:, 0]
            
        return states.T


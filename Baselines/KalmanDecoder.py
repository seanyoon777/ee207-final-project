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
        self.pred_dims = pred_dims
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
        X = np.ascontiguousarray(X.astype('float32'))
        Z = np.ascontiguousarray(Z.astype('float32'))
        
        # calculate the transition matrix from X_t to X_{t+1}
        X1 = X[:, :-1]
        X2 = X[:, 1:]
        A = lstsq_optimized(X1 @ X1.T, X1 @ X2.T).T 
        print("Transition matrix calculated")
        
        # calculate covariance of transition matrix
        residual = X2 - A @ X1
        W = residual @ residual.T / X1.shape[1] / C
        print("Covariance matrix calculated")

        # calculates the measurement matrix from X_t to X_{t+1}
        # This linearly relates hand state to neural firing
        H = lstsq_optimized(X @ X.T, X @ Z.T).T
        print("Measurement matrix calculated")

        # calculate the covariance of the measurement matrix
        residual = Z - H @ X
        Q = residual @ residual.T / X.shape[1]
        print("Measurement of covariance matrix calculated")
        
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
        X = np.ascontiguousarray(y_train.T.astype('float32'))
        Z = np.ascontiguousarray(X_train.T.astype('float32'))
        
        # compute matrices
        A, W, H, Q = self.update(X, Z, self.C)
        self.model = [A, W, H, Q] 


    def predict(self, X_test, y_init): 
        A, W, H, Q = self.model 
        y_test = self._predict(X_test, y_init, self.pred_dims, A, W, H, Q)
        return y_test 
    

    # @staticmethod
    # @nb.jit(nopython=True)
    def _predict(self, X_test, y_init, n_states, A, W, H, Q): 
        # transpose and make data contiguous 
        Z = np.ascontiguousarray(X_test.astype('float32'))

        # initialize 
        n_samples = Z.shape[1]
        states = np.zeros(shape=(n_states, n_samples), dtype='float32')  # state predictions
        P_m    = np.zeros(shape=(n_states, n_states), dtype='float32')   # prior error covariance
        P      = np.zeros(shape=(n_states, n_states), dtype='float32')   # posterior error covariance
        if y_init is None: 
            state  = np.ones(shape=(n_states, 1), dtype='float32')          # initial state as zeros
        else: 
            state  = y_init

        # make them contiguous 
        states = np.ascontiguousarray(states.astype('float32'))
        P_m    = np.ascontiguousarray(P_m.astype('float32'))
        P      = np.ascontiguousarray(P.astype('float32'))
        state  = np.ascontiguousarray(state.astype('float32'))

        # kalman filter loop 
        for t in tqdm(range(1, n_samples)): 
            # predict state and covariance 
            state_m = A @ state 
            P_m = A @ P @ A.T + W 
            
            # calculate system uncertainty and kalman gain
            S = H @ P_m @ H.T + Q 
            K = P_m @ H.T @ np.linalg.pinv(S)

            # update and store the updated state
            state = state_m + K @ (Z[:, t] - H @ state_m)
            P     = (np.eye(n_states) - K @ H) @ P_m 
            
            states[:, t] = state[:, 0]
            
        return states.T

@nb.jit(nopython=True) 
def lstsq_optimized(X: np.ndarray, y: np.ndarray):
    """Optimized jit-compatible helper function for calculating the Moore-Penrose pseudoinverse, 
    given by A+ = (AT A)-1 AT. Used for LRR calculation"""
    W = np.linalg.solve(X.T @ X, X.T @ y) 
    return W
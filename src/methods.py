import numpy as np
from quantes.linear import low_dim
from sklearn.linear_model import LinearRegression

from abc import ABC, abstractmethod

class ForcedSamplingBandit(ABC):
    '''
    Abstract base class for Bandit algorithms with forced sampling and all-sample estimators.
    '''
    @abstractmethod
    def __init__(self, q, h, d, K, **kwargs):
        '''
        Initialize the Bandit algorithm.

        Parameters
        ----------
        q : int
            Number of forced samples per arm in each round
        h : float
            Difference between optimal and suboptimal arms
        d : int
            Dimension of the context vectors
        K : int
            Number of arms
        '''
        pass
    
    @abstractmethod
    def choose_a(self, t, x):
        '''
        Choose an action based on the current time step and context vector.

        Parameters
        ----------
        t : int
            The current time step.
        x : np.ndarray
            The context vector of dimension d.

        Returns
        -------
        int
            The index of the chosen action.
        '''
        pass

    @abstractmethod
    def update_beta(self, rwd, t):
        '''
        Update the estimators based on the received reward.

        Parameters
        ----------
        rwd : float
            The reward received after taking the action.
        t : int
            The current time step.

        Returns
        -------
        None
        '''
        pass




class RiskAwareBandit(ForcedSamplingBandit):
    '''
    A class for the Risk-Aware Bandit algorithm with forced sampling and all-sample estimators.
    '''
    def __init__(self, q, h, tau, d, K, beta_real_value, alpha_real_value):
        '''
        Initialize the Risk-Aware Bandit algorithm.

        Parameters:
        ----------
        q (int) : Number of forced samples per arm in each round
        h (float) : Difference between optimal and suboptimal arms
        tau (float) : Quantile level for quantile regression
        d (int) : Dimension of the context vectors
        K (int) : Number of arms
        beta_real_value (np.ndarray) : True coefficient values for each arm
        alpha_real_value (np.ndarray) : True intercept values for each arm
        '''
        self.Tx = [[] for _ in range(K)]
        self.Sx = [[] for _ in range(K)]
        self.Tr = [[] for _ in range(K)]
        self.Sr = [[] for _ in range(K)]

        self.q = q
        self.h = h
        self.tau = tau
        self.d = d
        self.K = K

        self.set = np.array([])
        self.action = None  # Initialize action as well

        self.beta_t = np.random.uniform(0., 2., (K, d)) # forced sample estimator
        self.beta_a = np.random.uniform(0., 2., (K, d)) # all sample estimator
        self.alpha_t = np.random.uniform(0., 2., K) # forced sample intercept
        self.alpha_a = np.random.uniform(0., 2., K) # all sample intercept
        self.n = 0

        self.beta_real_value = beta_real_value
        self.alpha_real_value = alpha_real_value

        self.beta_error_a = np.zeros(K)
        self.beta_error_t = np.zeros(K)

    
    def choose_a(self,t,x): 
        """ 
        Choose an action based on the current time step and context vector.
        
        If the current time step is part of the forced sampling set,
        select the corresponding action. Otherwise, use the estimators
        to choose the action with the highest estimated reward within
        the acceptable range.
        Otherwise, select the action with the highest estimated reward.

        Parameters
        ----------
        t : int
            The current time step.
        x : np.ndarray
            The context vector of dimension d.
        Returns
        -------
        int
            The index of the chosen action.
        """
        # if t is the first time of the new round
        if t == ((2**self.n - 1)*self.K*self.q + 1):
            self.set = np.arange(t, t+self.q*self.K)
            self.n += 1

        if t in self.set: 
            ind = list(self.set).index(t)
            self.action = int(ind // self.q)
            self.Tx[self.action].append(x)
            self.Sx[self.action].append(x)
        else:
            forced_est = np.dot(self.beta_t, x) + self.alpha_t
            max_forced_est = np.amax(forced_est)
            K_hat = np.where(forced_est > max_forced_est - self.h/2.)[0]
            all_est = [np.dot(self.beta_a[k_hat], x) + self.alpha_a[k_hat] for k_hat in K_hat]
            self.action = K_hat[np.argmax(all_est)]
            self.Sx[self.action].append(x)

        # self.Sx[self.action].append(x)

        return self.action
    
    def update_beta(self, rwd, t):
        """ Update the estimators based on the received reward.
        For the first d samples, random initialization is used.
        """
        # Now check if we have enough samples to fit
        if np.array(self.Tx[self.action]).shape[0] > self.d:
            # Fit forced sampling estimator if we're in forced sampling set
            if t in self.set:
                self.Tr[self.action].append(rwd)
                forced_qr = low_dim(np.array(self.Tx[self.action]), 
                                    np.array(self.Tr[self.action]), 
                                    intercept=True).fit(tau=self.tau)
                self.beta_t[self.action] = forced_qr['beta'][1:]
                self.alpha_t[self.action] = forced_qr['beta'][0]

            # Always fit all-sample estimator
            self.Sr[self.action].append(rwd)
            all_qr = low_dim(np.array(self.Sx[self.action]), 
                            np.array(self.Sr[self.action]), 
                            intercept=True).fit(tau=self.tau)
            self.beta_a[self.action] = all_qr['beta'][1:]
            self.alpha_a[self.action] = all_qr['beta'][0]
        else:
            if t in self.set:
                self.Tr[self.action].append(rwd)
                forced_qr = low_dim(np.array(self.Tx[self.action]), 
                                    np.array(self.Tr[self.action]), 
                                    intercept=True).fit(tau=self.tau)
                self.beta_t[self.action] = forced_qr['beta'][1:]
                self.alpha_t[self.action] = forced_qr['beta'][0]
            self.Sr[self.action].append(rwd)
            self.beta_t[self.action] = np.random.uniform(0., 2., self.d)
            self.beta_a[self.action] = np.random.uniform(0., 2., self.d)
            self.alpha_t[self.action] = np.random.uniform(0., 2.)
            self.alpha_a[self.action] = np.random.uniform(0., 2.)
        
        # Always update errors
        self.beta_error_a[self.action] = np.linalg.norm(
            self.beta_a[self.action] - self.beta_real_value[self.action]
        )
        self.beta_error_t[self.action] = np.linalg.norm(
            self.beta_t[self.action] - self.beta_real_value[self.action]
        )

class OLSBandit(ForcedSamplingBandit):
    def __init__(self, q, h, d, K, beta_real_value):
        """Initialize the OLSBandit with given parameters.
        """
        self.Sx = [[] for _ in range(K)]
        self.Sr = [[] for _ in range(K)]
        self.Tx = [[] for _ in range(K)]
        self.Tr = [[] for _ in range(K)]

        self.beta_a = np.random.uniform(0., 2., (K, d))
        self.beta_t = np.random.uniform(0., 2., (K, d))

        self.beta_real_value = beta_real_value

        self.beta_error_a = np.zeros(K)
        self.beta_error_t = np.zeros(K)

        self.q = q
        self.h = h
        self.d = d
        self.K = K

        self.n = 0

        self.set = np.array([])
        self.action = None  # Initialize action as well

    def choose_a(self, t, x):
        """Choose an action based on the current time step and context vector.
        """
        if t == ((2**self.n - 1)*self.K*self.q + 1):
            self.set = np.arange(t, t+self.q*self.K)
            self.n += 1
        if t in self.set:
            ind = list(self.set).index(t)
            self.action = int(ind // self.q)
            self.Tx[self.action].append(x)
        else:
            # no intercept
            forced_est = np.dot(self.beta_t, x)
            max_forced_est = np.amax(forced_est)
            K_hat = np.where(forced_est > max_forced_est - self.h/2.)[0]
            all_est = [np.dot(self.beta_a[k_hat], x) for k_hat in K_hat]
            self.action = int(K_hat[np.argmax(all_est)])
            
        self.Sx[self.action].append(x)

        return self.action


    def update_beta(self, rwd, t):
        """Update the estimators based on the received reward.
        """
        if np.array(self.Tx[self.action]).shape[0] > self.d:
            if t in self.set:
                self.Tr[self.action].append(rwd)
                forced_ols = LinearRegression(fit_intercept=False)
                forced_ols.fit(np.array(self.Tx[self.action]), np.array(self.Tr[self.action]))
                self.beta_t[self.action] = forced_ols.coef_

            self.Sr[self.action].append(rwd)
            all_ols = LinearRegression(fit_intercept=False)
            all_ols.fit(np.array(self.Sx[self.action]), np.array(self.Sr[self.action]))
            self.beta_a[self.action] = all_ols.coef_

            self.beta_error_a[self.action] = np.linalg.norm(self.beta_a[self.action] - self.beta_real_value[self.action])
            self.beta_error_t[self.action] = np.linalg.norm(self.beta_t[self.action] - self.beta_real_value[self.action])
        else:
            if t in self.set:
                self.Tr[self.action].append(rwd)
                forced_ols = LinearRegression(fit_intercept=False)
                forced_ols.fit(np.array(self.Tx[self.action]), np.array(self.Tr[self.action]))
                self.beta_t[self.action] = forced_ols.coef_
            self.Sr[self.action].append(rwd)
            # self.Tr[self.action].append(rwd)
            self.beta_t[self.action] = np.random.uniform(0., 2., self.d)
            self.beta_a[self.action] = np.random.uniform(0., 2., self.d)

            self.beta_error_a[self.action] = np.linalg.norm(self.beta_a[self.action] - self.beta_real_value[self.action])
            self.beta_error_t[self.action] = np.linalg.norm(self.beta_t[self.action] - self.beta_real_value[self.action])

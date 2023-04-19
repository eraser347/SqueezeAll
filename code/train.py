import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve, minimize
from sklearn.linear_model import LogisticRegression
import sobol_seq ## For quasi-Monte carlo estimation

## For quick update of Vinv
def sherman_morrison(X, V, w=1):
    result = V-(w*np.einsum('ij,j,k,kl -> il', V, X, X, V))/(1.+w*np.einsum('i,ij,j ->', X, V, X))
    return result


'''
TS
'''
class TS:
    def __init__(self, d, v):
        ## Initialization
        self.beta_hat=np.zeros(d)
        self.f=np.zeros(d)
        self.Binv=np.eye(d)
        self.t = 0

        ## Hyperparameters
        self.v=v
        self.settings = {'v': self.v}

    def select_ac(self,contexts):
        ## Sample beta_tilde.
        N=len(contexts)
        V=(self.v**2)*self.Binv
        beta_tilde=np.random.multivariate_normal(self.beta_hat, V, size=N)
        est=np.array([np.dot(contexts[i], beta_tilde[i,]) for i in range(N)])
        ## Selecting action with tie-breaking.
        a_t=np.argmax(est)
        self.X_a=contexts[a_t]
        return(a_t)

    def update(self,reward):
        self.f=self.f+reward*self.X_a
        self.Binv = sherman_morrison(X=self.X_a, V=self.Binv)
        self.beta_hat=np.dot(self.Binv, self.f)


'''
HyRan
'''
class HyRan:
    def __init__(self, d, p, lam=1):
        ##Initialization
        self.t=0
        self.d=d
        self.yx=np.zeros(d)
        self.f=np.zeros(d)
        self.Vinv=lam*np.eye(d)
        self.Binv=lam*np.eye(d)
        self.V = np.zeros((d,d))
        self.W = np.zeros((d,d))
        self.beta_hat=np.zeros(d)
        self.beta_tilde=np.zeros(d)
        self.ridgeVinv = np.eye(d)
        self.ridgef = np.zeros(d)
        self.psi_size = 1 

        ## Hyperparameters
        self.p = p
        self.lam = lam
        self.settings = {'lam':lam, 'p':p}

    def select_ac(self, contexts):
        # contexts: list [X(1),...X(N)]
        self.t = self.t + 1
        self.lam = 2*self.d*np.log(self.t+1)
        N = len(contexts)
        means = np.array([np.dot(X, self.beta_hat) for X in contexts])
        a_t = np.argmax(means)
        pi = (1-self.p)/(N-1)*np.ones(N)
        pi[a_t] = self.p
        tilde_a_t = np.argmax(np.random.multinomial(1, pi, size=1))
        self.DR = (tilde_a_t == a_t)
       ## Update matrices
        if self.DR: #when tilde_a_t == a_t
            X = np.array(contexts)
            self.V = self.V + X.T @ X
            self.psi_size = self.psi_size + 1
            try:
                self.Vinv = np.linalg.inv(self.V + self.lam*np.eye(self.d))
                self.Binv = np.linalg.inv(self.V + N*self.lam*np.sqrt(self.psi_size)*self.lam*np.eye(self.d))
            except:
                for i in range(N):
                    self.Vinv = sherman_morrison(contexts[i], self.Vinv)
                    self.Binv = sherman_morrison(contexts[i], self.Binv)
            self.W = self.W + np.outer(contexts[a_t], contexts[a_t]) / self.p
            
        else: #when tilde_a_t != a_t
            self.Vinv = sherman_morrison(contexts[a_t], self.Vinv)
        self.X_a = contexts[a_t]
        return(a_t)

    def update(self, reward):
        # Update beta_tilde
        self.ridgeVinv = sherman_morrison(self.X_a, self.ridgeVinv)
        self.ridgef = self.ridgef + reward * self.X_a
        ridge = self.ridgeVinv @ self.ridgef
        if np.linalg.norm(ridge) > 1:
            ridge = ridge/np.linalg.norm(ridge)

        # Update beta_tilde and beta_hat
        if self.DR: #when tilde_a_t == a_t
            self.f = self.f + reward*self.X_a/self.p
            self.beta_tilde = ridge + self.Vinv @ (-self.lam*ridge - self.W @ ridge + self.f)
            self.beta_hat = self.beta_tilde + self.Vinv @ (-self.lam*self.beta_tilde - self.W @ self.beta_tilde + self.f)
        else: #when tilde_a_t != a_t
            self.f = self.f + reward*self.X_a
            self.beta_tilde = self.Binv @ self.f
            self.beta_hat = self.Vinv @ self.f




'''
UCB
'''
class UCB:
    def __init__(self, d, alpha, lam=1):
        self.alpha=alpha
        self.d=d
        self.yx=np.zeros(d)
        self.Binv=lam*np.eye(d)
        self.beta_hat = np.zeros(d)
        self.settings = {'alpha': self.alpha}

    def select_ac(self, contexts):
        means = np.array([np.dot(X, self.beta_hat) for X in contexts])
        stds = np.array([np.sqrt(X.T @ self.Binv @ X) for X in contexts])
        ucbs = means + self.alpha*stds
        a_t = np.argmax(ucbs)
        self.X_a = contexts[a_t]
        return(a_t)

    def update(self,reward):
        self.Binv = sherman_morrison(self.X_a, self.Binv)
        self.yx = self.yx+reward*self.X_a
        self.beta_hat = self.Binv @ self.yx


'''
SuplinUCB
'''
class SuplinUCB:
    def __init__(self, d, alpha, T, lam=1):
        self.alpha=alpha
        self.d=d
        self.T=T
        self.S=int(np.log(T))+1
        self.yxs=[np.zeros(d) for s in range(self.S)]
        self.Binvs=[lam*np.eye(d) for s in range(self.S)]
        self.beta_hats = [np.zeros(d) for s in range(self.S)]
        self.settings = {'alpha': self.alpha}

    def select_ac(self, contexts):
        indexes = [True for i in range(len(contexts))]
        self.s_index = -1 #initializing index for updates: (-1 -> exploitation)
        for s in range(1, self.S+1):
            means = np.array([np.dot(X, self.beta_hats[s-1]) for X in contexts])
            widths = self.alpha*np.array([np.sqrt(X.T @ self.Binvs[s-1] @ X) for X in contexts])
            ucbs = means + widths
            if np.max(widths[indexes]) <= 1/np.sqrt(self.T): #exploitation
                a_t = np.argmax(ucbs[indexes])
                self.X_a = contexts[a_t]
                break
            elif np.max(widths[indexes]) <= 2**(-s): #elimination
                indexes = [ucbs >= np.max(ucbs[indexes]) - 2**(1-s)]
            else: #exploration
                a_t = np.argmax(widths[indexes]) 
                self.X_a = contexts[a_t]
                self.s_index = s-1
                break
        return(a_t)

    def update(self,reward): #Update the s-th self.Binv
        if self.s_index >= 0:
            self.Binvs[self.s_index] = sherman_morrison(self.X_a, self.Binvs[self.s_index])
            self.yxs[self.s_index] = self.yxs[self.s_index]+reward*self.X_a
            self.beta_hats[self.s_index] = self.Binvs[self.s_index] @ self.yxs[self.s_index]



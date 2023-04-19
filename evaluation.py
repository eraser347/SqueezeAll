import numpy as np
from tqdm import trange
from train import HyRan, UCB, SuplinUCB, TS
from data import generate_contexts
import time, json, itertools

def eval_UCB(N, d, alpha_set=[0.001, 0.01, 0.1, 1], T=30000, M=10, rho=0.5, R=1, seed=0):
    #evaluate UCB
    #inputs: M, N, d, T, rho, seed, B(bound for the beta)
    results = []
    for alpha in alpha_set:
        cumul_regret = np.zeros((M,T))
        beta_err = np.zeros((M,T))
        elapsed_time = np.zeros((M,T))
        for m in range(M):
            print('UCB Simulation %d, N=%d, d=%d, alpha=%.3f' % (m+1, N, d, alpha))
            # call model
            M_UCB = UCB(d=d, alpha=alpha)
            # true beta
            np.random.seed(seed+m)
            #beta = np.random.uniform(-1,1,d)
            beta = np.random.uniform(-1/np.sqrt(d),1/np.sqrt(d),d)
            opt_reward = []
            UCB_reward = []

            for t in trange(T):
                # generate contexts
                contexts = generate_contexts(N, d, rho, seed=seed+t+m)
                # optimal reward
                opt_reward.append(np.amax(np.array(contexts) @ beta))
                # time
                start = time.time()
                a_t = M_UCB.select_ac(contexts)
                reward = np.dot(contexts[a_t],beta) + np.random.normal(0, R, size=1)
                UCB_reward.append(np.dot(contexts[a_t],beta))
                M_UCB.update(reward)
                elapsed_time[m,t] = time.time() - start
                beta_err[m,t] = np.linalg.norm(M_UCB.beta_hat-beta)

            cumul_regret[m,:] = np.cumsum(opt_reward)-np.cumsum(UCB_reward)
        ##Save at dict
        results.append({'model':'UCB',
                        'settings':M_UCB.settings,
                        'regrets':cumul_regret.tolist(),
                        'beta_err':beta_err.tolist(),
                        'time':elapsed_time.tolist()})
    ##Save to txt file
    with open('./results/UCB_d%d_N%d.txt' % (d, N), 'w') as outfile:
        json.dump(results, outfile)



def eval_SuplinUCB(N, d, alpha_set=[0.001, 0.01, 0.1, 1], T=30000, M=10, rho=0.5, R=1, seed=0):
    #evaluate SuplinUCB
    #inputs: M, N, d, T, rho, seed
    results = []
    for alpha in alpha_set:
        cumul_regret = np.zeros((M,T))
        beta_err = np.zeros((M,T))
        elapsed_time = np.zeros((M,T))
        for m in range(M):
            print('SuplinUCB Simulation %d, N=%d, d=%d, alpha=%.3f' % (m+1, N, d, alpha))
            # call model
            M_UCB = SuplinUCB(d=d, alpha=alpha, T=T)
            # true beta
            np.random.seed(seed+m)
            #beta = np.random.uniform(-1,1,d)
            beta = np.random.uniform(-1/np.sqrt(d),1/np.sqrt(d),d)
            opt_reward = []
            UCB_reward = []

            for t in trange(T):
                # generate contexts
                contexts = generate_contexts(N, d, rho, seed=seed+t+m)
                # optimal reward
                opt_reward.append(np.amax(np.array(contexts) @ beta))
                # time
                start = time.time()
                a_t = M_UCB.select_ac(contexts)
                reward = np.dot(contexts[a_t],beta) + np.random.normal(0, R, size=1)
                UCB_reward.append(np.dot(contexts[a_t],beta))
                M_UCB.update(reward)
                elapsed_time[m,t] = time.time() - start
                #beta_err[m,t] = np.linalg.norm(M_UCB.beta_hat-beta)

            cumul_regret[m,:] = np.cumsum(opt_reward)-np.cumsum(UCB_reward)
        ##Save at dict
        results.append({'model':'SuplinUCB',
                        'settings':M_UCB.settings,
                        'regrets':cumul_regret.tolist(),
                        #'beta_err':beta_err.tolist(),
                        'time':elapsed_time.tolist()})
    ##Save to txt file
    with open('./results/SuplinUCB_d%d_N%d.txt' % (d, N), 'w') as outfile:
        json.dump(results, outfile)



def eval_TS(N, d, v_set=[0.001, 0.01, 0.1, 1], T=30000, M=10, rho=0.5, R=1, seed=0):
    #evaluate UCB
    #inputs: M, N, d, T, rho, seed, B(bound for the beta)
    results = []
    for v in v_set:
        cumul_regret = np.zeros((M,T))
        beta_err = np.zeros((M,T))
        elapsed_time = np.zeros((M,T))
        for m in range(M):
            print('TS Simulation %d, N=%d, d=%d, v=%.3f' % (m+1, N, d, v))
            # call model
            M_TS = TS(d=d, v=v)
            # true beta
            np.random.seed(seed+m)
            #beta = np.random.uniform(-1,1,d)
            beta = np.random.uniform(-1/np.sqrt(d),1/np.sqrt(d),d)
            opt_reward = []
            UCB_reward = []

            for t in trange(T):
                # generate contexts
                contexts = generate_contexts(N, d, rho, seed=seed+t+m)
                # optimal reward
                opt_reward.append(np.amax(np.array(contexts) @ beta))
                # time
                start = time.time()
                a_t = M_TS.select_ac(contexts)
                reward = np.dot(contexts[a_t],beta) + np.random.normal(0, R, size=1)
                UCB_reward.append(np.dot(contexts[a_t],beta))
                M_TS.update(reward)
                elapsed_time[m,t] = time.time() - start
                beta_err[m,t] = np.linalg.norm(M_TS.beta_hat-beta)

            cumul_regret[m,:] = np.cumsum(opt_reward)-np.cumsum(UCB_reward)
        ##Save at dict
        results.append({'model':'TS',
                        'settings':M_TS.settings,
                        'regrets':cumul_regret.tolist(),
                        'beta_err':beta_err.tolist(),
                        'time':elapsed_time.tolist()})
    ##Save to txt file
    with open('./results/TS_d%d_N%d.txt' % (d, N), 'w') as outfile:
        json.dump(results, outfile)


def eval_HyRan(N, d, p_set=[0.5, 0.65, 0.8, 0.95], T=30000, M=10, rho=0.5, R=1, seed=0):
    #evaluate HyRan
    results = []
    for p in p_set:
        cumul_regret = np.zeros((M,T))
        beta_err = np.zeros((M,T))
        elapsed_time = np.zeros((M,T))
        for m in range(M):
            print('HyRan Simulation %d, N=%d, d=%d, p=%.2f' % (m+1, N, d, p))
            # call model
            M_HyRan = HyRan(d=d, p=p, lam=1)
            # true beta
            np.random.seed(seed+m)
            #beta = np.random.uniform(-1,1,d)
            beta = np.random.uniform(-1/np.sqrt(d),1/np.sqrt(d),d)
            opt_reward = []
            UCB_reward = []

            for t in trange(T):
                # generate contexts
                contexts = generate_contexts(N, d, rho, seed=seed+t+m)
                # optimal reward
                opt_reward.append(np.amax(np.array(contexts) @ beta))
                # time
                start = time.time()
                a_t = M_HyRan.select_ac(contexts)
                reward = np.dot(contexts[a_t],beta) + np.random.normal(0, R, size=1)
                UCB_reward.append(np.dot(contexts[a_t],beta))
                M_HyRan.update(reward)
                elapsed_time[m,t] = time.time() - start
                beta_err[m,t] = np.linalg.norm(M_HyRan.beta_hat-beta)

            cumul_regret[m,:] = np.cumsum(opt_reward)-np.cumsum(UCB_reward)
        ##Save at dict
        results.append({'model':'HyRan',
                        'settings':M_HyRan.settings,
                        'regrets':cumul_regret.tolist(),
                        'beta_err':beta_err.tolist(),
                        'time':elapsed_time.tolist()})
    ##Save to txt file
    with open('./results/HyRan_d%d_N%d.txt' % (d, N), 'w') as outfile:
        json.dump(results, outfile)



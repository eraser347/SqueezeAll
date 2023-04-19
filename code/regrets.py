import numpy as np
import matplotlib.pyplot as plt
import json, itertools


T=30000

#for d,N in itertools.product([5],[5]):
plt.figure(figsize=(30,20))
i=1
for d,N in itertools.product([5,10,20],[10,20]):
    with open('./results/DRUCB_d%d_N%d.txt' % (d, N)) as infile:
        results_DR = json.load(infile)
    with open('./results/UCB_d%d_N%d.txt' % (d, N)) as infile:
        results_UCB = json.load(infile)
    with open('./results/TS_d%d_N%d.txt' % (d, N)) as infile:
        results_TS = json.load(infile)
    with open('./results/SuplinUCB_d%d_N%d.txt' % (d, N)) as infile:
        results_SuplinUCB = json.load(infile)

    last_regret = []
    for result in results_UCB:
        alpha = result['settings']['alpha']
        last_regret.append(np.mean(result['regrets'], axis=0)[-1])
    best_UCB = np.argmin(last_regret)

    last_regret = []
    for result in results_DR:
        alpha = result['settings']['alpha']
        last_regret.append(np.mean(result['regrets'], axis=0)[-1])
    best_DR = np.argmin(last_regret)

    last_regret = []
    for result in results_TS:
        v = result['settings']['v']
        last_regret.append(np.mean(result['regrets'], axis=0)[-1])
    best_TS = np.argmin(last_regret)
    
    last_regret = []
    for result in results_SuplinUCB:
        v = result['settings']['alpha']
        last_regret.append(np.mean(result['regrets'], axis=0)[-1])
    best_SuplinUCB = np.argmin(last_regret)

    plt.subplot(4,4,i)
    plt.plot(np.arange(1,T+1), np.mean(results_DR[best_DR]['regrets'], axis=0), label='DRUCB')
    plt.plot(np.arange(1,T+1), np.mean(results_UCB[best_UCB]['regrets'], axis=0), '-.', label='UCB')
    #plt.plot(np.arange(1,T+1), np.mean(results_SuplinUCB[best_SuplinUCB]['regrets'], axis=0), '-.', label='SuplinUCB')
    plt.plot(np.arange(1,T+1), np.mean(results_TS[best_TS]['regrets'], axis=0), '-.', label='TS')
    #plt.ylim(0,180)
    plt.title('N=%d, d=%d' % (N,d))
    #plt.xlabel('t:Number of rounds')
    #plt.ylabel('Regrets')
    i = i+1
    '''
    plt.figure()
    plt.plot(np.arange(1,T+1), np.mean(results_DR[best_DR]['beta_err'], axis=0), label='DRUCB')
    plt.plot(np.arange(1,T+1), np.mean(results_UCB[best_UCB]['beta_err'], axis=0), '--', label='UCB')
    plt.plot(np.arange(1,T+1), np.mean(results_TS[best_TS]['beta_err'], axis=0), '--', label='TS')
    plt.legend(title='N=%d, d=%d' % (N,d))
    plt.xlabel('Decision points')
    plt.ylabel('beta_err')
    plt.savefig('./errs_N_%d_d_%d.pdf' % (N,d), bbox_inches='tight', pad_inches=0, dpi=50)
    plt.close()
    '''
plt.savefig('./regrets.pdf', bbox_inches='tight', pad_inches=0, dpi=50)
plt.close()

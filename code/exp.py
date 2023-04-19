from evaluation import eval_UCB, eval_HyRan, eval_SuplinUCB, eval_TS
for d in [5, 10, 20]:
    for N in [10, 20]:
        #eval_UCB(N=N, d=d)
        #eval_TS(N=N, d=d)
        #eval_HyRan(N=N, d=d)
        eval_SuplinUCB(N=N, d=d)

# Squeeze All: Novel Estimator and Self-Normalized Bound for Linear Contextual Bandits

Python3 based implementation of the paper "# Squeeze All: Novel Estimator and Self-Normalized Bound for Linear Contextual Bandits".

## Directory tree

```bash
├────── results
├── data.py
├── evaluation.py
├── exp.py
├── regrets.py
└── train.py
```
- `data.py` contains the codes for generating contexts.
- `evaluation.py` evaluates the four algorithms in the paper.
- `exp.py` to generates the results and save it to `./results`.
- `regrets.py` plots the cumulative regrets with the data saved in `./results`
- To obtain the results, enter `python3 exp.py` and then run `python3 regrets.py`

## Requirements
- python 3
- numpy
- scipy
- matplotlib
- tqdm

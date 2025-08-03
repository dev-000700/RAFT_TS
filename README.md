# Retrieval-Augmented Forecasting of Time-series (RAFT)
This is the official PyTorch implementation of our paper ([Link](https://arxiv.org/abs/2505.04163)), which is accepted to ICML 2025. \
The code is build on the base of [Time-Series-Library](https://github.com/thuml/Time-Series-Library).


### Required Packages
* python == 3.9.13
* numpy == 1.24.3
* torch == 1.10.0
* tqdm == 4.65.0

### Usage
1. Create ./data directory and place dataset files in ./data directory.
2. Run following code.
```
pip install sktime
python3 run.py --data GOOG --data_path GOOG.csv --seq_len 10 --pred_len 10 --top_k 3

```
python3 run.py --data ETTh1 --top_k 3 --data_path ETTh1.csv --root_path ./data/ETT
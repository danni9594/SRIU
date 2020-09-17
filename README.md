# SRIU
This is our experimental code for Recommender System Incremental Update via Sample Reweighting (SRIU). 

# Data Preprocessing
[Tmall Dataset](https://tianchi.aliyun.com/dataset/dataDetail?dataId=42) `data_format1 > user_log_format1.csv`\
[Sobazaar Dataset](https://github.com/hainguyen-telenor/Learning-to-rank-from-implicit-feedback) `Data > Sobazaar-hashID.csv.gz`\
To preprocess the above raw dataset, save them in the `raw_datasets` directory under the root directory, navigate to the `data_preprocess` directory and do `python tmall_preprocess.py` or `python soba_preprocess.py`. The preprocessed datasets will be saved in the `datasets` directory for later used.

# Periodic Training
We compare SRIU with the other 3 training strategies. To perform periodic training using any of the training strategy, navigate to the repsective directory, and do `python train_tmall.py` or `python train_soba.py`. Hyper-paramemeters can be configured conveniently in the entry files.

# Visualize Weights
For SRIU, our code generates and saves the computed sample weights for every training period. Here, we use the `weight.csv` from the last period of Tmall dataset for visualization, presented in `visualize_weights.ipynb`.

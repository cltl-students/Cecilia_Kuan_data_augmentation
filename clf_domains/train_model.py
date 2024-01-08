"""
code adapted and modified from A-PROOF repository https://github.com/cltl/a-proof-zonmw
==============================

This is part 2 of a dual-classifiers fine-tuning approach.
Fine-tune and save a multi-label classification model with Simple Transformers.

The script can be customized with the following parameters:
    --train_pkl: the file with the train data
    --eval_pkl: the file with the eval data
    --model_args: configurations used
    --model_type: type of the pre-trained model, e.g. bert, roberta, electra
    --model_name: the pre-trained model, either from Hugging Face or locally stored
    --gold_col: column name of gold labels
"""


import logging
import warnings
import torch
import pandas as pd
from simpletransformers.classification import MultiLabelClassificationModel

import sys
sys.path.insert(0, '..')

import os
os.environ['WANDB_MODE'] = 'offline'

def train(
    train_pkl,
    eval_pkl,
    model_args,
    model_type,
    model_name,
    gold_col,
    labels=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'],
):
    """
    Fine-tune and save a multi-label classification model with Simple Transformers.

    Parameters
    ----------
    train_pkl: str
        path to pickled df with the training data, which must contain the columns 'text' and 'labels'; the labels are multi-hot lists (see column indices in `labels`), e.g. [1, 0, 0, 1, 0, 0, 0, 0, 1]
    eval_pkl: {None, str}
        path to pickled df for evaluation during training (optional)
    model_args: str
        configurations used for fine-tuning
    model_type: str
        type of the pre-trained model, e.g. bert, roberta, electra
    model_name: str
        the exact architecture and trained weights to use; this can be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model file
    labels: list
        list of column indices for the multi-hot labels
    gold_col: str
        column name of gold labels

    Returns
    -------
    None
    """

    # check CUDA
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        def custom_formatwarning(msg, *args, **kwargs):
            return str(msg) + '\n'
        warnings.formatwarning = custom_formatwarning
        warnings.warn(' ====== CUDA device not available; running on a CPU! ====== ')

    # logging
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger('transformers')
    transformers_logger.setLevel(logging.WARNING)

    # load data
    train_data = pd.read_pickle(train_pkl)
    train_data = train_data.rename({gold_col:'labels'}, axis=1)
    train_df = train_data[['text','labels']].copy()

    eval_data = pd.read_pickle(eval_pkl)
    eval_data = train_data.rename({gold_col:'labels'}, axis=1)
    eval_df = eval_data[['text','labels']].copy()

    # model
    model = MultiLabelClassificationModel(
        model_type,
        model_name,
        num_labels=len(labels),
        args=model_args,
        use_cuda=cuda_available,
    )

    # train
    if model.args.evaluate_during_training:
        model.train_model(train_df, eval_df=eval_df)
    else:
        print(" ====== Training starts ====== ")
        model.train_model(train_df)


if __name__ == '__main__':

    train_pkl = '../data/combined/train_gpt-anon_jenia-pos-only_all-labels.pkl'
    eval_pkl = '../data/jenia/dev_jenia_all-labels.pkl'
    model_type = 'roberta'
    model_name = '../models/medroberta' 
    gold_col = 'labels_9'

    model_args = {
        "max_seq_length": 512,
        "output_dir": "../models/mR_gpt-anon_jenia-pos_e5b",
        "best_model_dir": "../models/mR_gpt-anon_jenia-pos_e5b/best_model/",
        "tensorboard_dir": "../models/runs/",
        "cache_dir": "../models/cache_dir/",
        "dataloader_num_workers": 12,
        "process_count":12,
        "use_multiprocessing": False,
        "use_multiprocessing_for_evaluation": False,
        "silent": False,
        "manual_seed": 19,
        "num_train_epochs": 2
        } 

    train(
        train_pkl,
        eval_pkl,
        model_args,
        model_type,
        model_name,
        gold_col
    )

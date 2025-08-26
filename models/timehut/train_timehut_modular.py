#!/usr/bin/env python3
"""
TimeHUT Training Script - Modular Version
Uses the modular TimeHUT implementation from timehut_modules
"""

import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
import json
from sklearn.model_selection import train_test_split
import pyhopper

# Import modular TimeHUT components
from timehut_modules import TimeHUT
from timehut_tasks import eval_classification
from timehut_datautils import load_UCR, load_UEA
from timehut_utils import init_dl_program

labeled_ratio = 0.7


def train_model(config, temp_dictionary=None, type="full"):
    '''
    Trains the TimeHUT model using either full dataset or the split dataset
    '''
    
    if type == 'full':
        print("Training the final model")
        t = time.time()
        model = TimeHUT(
            input_dims=train_data.shape[-1],
            device=device,
            temp_dictionary=temp_dictionary,
            **config
        )
        loss_log = model.fit(
            train_data,
            n_epochs=args.epochs,
            n_iters=args.iters,
            verbose=False
        )
        
        if task_type == 'classification':
            out, eval_res = eval_classification(
                model, train_data, train_labels, test_data, test_labels, eval_protocol='svm'
            )
        else:
            raise NotImplementedError(f"Task type {task_type} not implemented in modular version")
            
        t = time.time() - t
        print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
        print(f'Evaluation result on test (full train): {eval_res}')
        
        return eval_res
    
    elif type == 'split':
        print("Training the split model")
        t = time.time()
        model = TimeHUT(
            input_dims=train_data.shape[-1],
            device=device,
            temp_dictionary=temp_dictionary,
            **config
        )
        loss_log = model.fit(
            train_data_split,
            n_epochs=args.epochs,
            n_iters=args.iters,
            verbose=False
        )
        t = time.time() - t
        
        out_val, eval_res_val = eval_classification(
            model, train_data_split, train_labels_split, 
            val_data_split, val_labels_split, eval_protocol='svm'
        )
        print('Evaluation result (val)               :', eval_res_val)
        
        out_test, eval_res_test = eval_classification(
            model, train_data_split, train_labels_split, 
            test_data, test_labels, eval_protocol='svm'
        )
        print('Evaluation result (test)              :', eval_res_test)
        
        return eval_res_val


def main():
    global train_data, train_labels, test_data, test_labels
    global train_data_split, val_data_split, train_labels_split, val_labels_split
    global task_type, device, args, out_name
    
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics')
    parser.add_argument('--loader', type=str, required=True, help='The data loader (UCR, UEA, etc.)')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension')
    parser.add_argument('--max-train-length', type=int, default=3000, help='Maximum training length')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save checkpoint frequency')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='Maximum allowed threads')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='Ratio of missing observations')
    parser.add_argument('--method', type=str, default='acc', help='Evaluation method (acc or auprc)')
    parser.add_argument('--dataroot', type=str, default='/media/milad/DATA/TSResearch/datasets', help='Dataset root path')
    
    args = parser.parse_args()
    
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    out_dir = "results/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    out_name = f"{out_dir}/{args.run_name}.json"
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    
    print("Loading data...")
    
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = load_UCR(args.dataset, root=args.dataroot)
        train_data_split, val_data_split, train_labels_split, val_labels_split = train_test_split(
            train_data, train_labels, test_size=labeled_ratio, random_state=101
        )
        
    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = load_UEA(args.dataset, root=args.dataroot)
        train_data_split, val_data_split, train_labels_split, val_labels_split = train_test_split(
            train_data, train_labels, test_size=labeled_ratio, random_state=101
        )
    else:
        raise ValueError(f"Unknown loader: {args.loader}")
    
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
    
    # Define hyperparameter optimization objective
    def objective(hparams: dict):
        temp_settings = {}
        temp_settings['min_tau'] = hparams['min_tau']
        temp_settings['max_tau'] = hparams['max_tau']
        temp_settings['t_max'] = hparams['t_max']
        out = train_model(config, temp_settings, type="split")
        return out['acc']
        
    # Hyperparameter search
    search = pyhopper.Search(
        {
            "min_tau": pyhopper.float(0, 0.3, "0.05f"),
            "max_tau": pyhopper.float(0.5, 1, "0.05f"),
            "t_max": pyhopper.float(1, 20, "0.5f"),
        }
    )
    
    temp_settings = search.run(objective, "maximize", steps=10, n_jobs=1)
    print("Best Params:", temp_settings)
    
    # Train final model with best parameters
    print("Training the final model with best parameters")
    final_res = train_model(config, temp_settings, type="full")

    # Save results
    output = {}
    output['best'] = temp_settings
    output['res'] = final_res
    
    with open(out_name, "w") as out_file:
        json.dump(output, out_file, indent=2)
    
    print("Finished.")


if __name__ == '__main__':
    main()

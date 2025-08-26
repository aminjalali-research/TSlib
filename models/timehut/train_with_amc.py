import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from ts2vec import TS2Vec
import tasks
import datautils
from utils import init_dl_program
from sklearn.model_selection import train_test_split
import pyhopper
import json

labeled_ratio = 0.7

def train_model_with_amc(config, temp_dictionary=None, amc_setting=None, type="full"):
    """Train TimeHUT model with proper AMC loss configuration"""
    
    if type == 'full':
        print("🔥 Training TimeHUT with AMC losses enabled")
        print(f"   🎯 AMC Instance: {amc_setting['amc_instance'] if amc_setting else 0}")
        print(f"   🎯 AMC Temporal: {amc_setting['amc_temporal'] if amc_setting else 0}")
        print(f"   🎯 AMC Margin: {amc_setting['amc_margin'] if amc_setting else 0.5}")
        
        t = time.time()
        model = TS2Vec(
            input_dims=train_data.shape[-1],
            device=device,
            temp_dictionary=temp_dictionary,
            amc_setting=amc_setting,  # ⚡ CRITICAL: Pass AMC settings
            **config
        )
        
        loss_log = model.fit(
            train_data,
            n_epochs=args.epochs,
            n_iters=args.iters,
            verbose=False
        )

        out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
        print('Evaluation result on test (full train):', eval_res)
        return eval_res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics')
    parser.add_argument('--loader', type=str, required=True, help='The data loader (UCR or UEA)')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension')
    parser.add_argument('--max-train-length', type=int, default=3000, help='Max train length')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum threads')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation')
    parser.add_argument('--dataroot', type=str, default='/home/amin/TSlib/datasets', help='Root for dataset')
    
    # ⚡ CRITICAL AMC Parameters
    parser.add_argument('--amc-instance', type=float, default=0.5, help='AMC instance coefficient')
    parser.add_argument('--amc-temporal', type=float, default=0.5, help='AMC temporal coefficient')  
    parser.add_argument('--amc-margin', type=float, default=0.5, help='AMC margin')
    
    # Temperature scheduling parameters
    parser.add_argument('--min-tau', type=float, default=0.15, help='Min temperature')
    parser.add_argument('--max-tau', type=float, default=0.75, help='Max temperature')
    parser.add_argument('--t-max', type=float, default=10.5, help='Temperature period')
    
    args = parser.parse_args()
    
    print("🔥 TimeHUT with AMC Losses Configuration:")
    print(f"   Dataset: {args.dataset}")
    print(f"   AMC Instance: {args.amc_instance}")  
    print(f"   AMC Temporal: {args.amc_temporal}")
    print(f"   AMC Margin: {args.amc_margin}")
    print(f"   Temperature range: {args.min_tau} - {args.max_tau}")
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    
    print('Loading data... ', end='')
    if args.loader == 'UCR':
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset, root=args.dataroot)
    elif args.loader == 'UEA':
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset, root=args.dataroot)
    print('done')
    
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
    
    # ⚡ CRITICAL: Configure AMC settings with NON-ZERO values
    amc_setting = {
        'amc_instance': args.amc_instance,  # Must be > 0 for Angular Margin Contrastive loss
        'amc_temporal': args.amc_temporal,  # Must be > 0 for temporal AMC loss
        'amc_margin': args.amc_margin       # Angular margin value
    }
    
    # Configure temperature settings  
    temp_settings = {
        'min_tau': args.min_tau,
        'max_tau': args.max_tau,
        't_max': args.t_max,
        'method': 'cosine_annealing'
    }
    
    print(f"✅ AMC Settings: {amc_setting}")
    print(f"✅ Temperature Settings: {temp_settings}")
    
    # Train with AMC losses enabled
    final_res = train_model_with_amc(config, temp_dictionary=temp_settings, amc_setting=amc_setting, type="full")
    
    print("✅ TimeHUT with AMC losses completed successfully!")

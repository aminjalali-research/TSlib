# TimesURL - Original Time Series Representation Learning

## Overview
Original TimesURL implementation for unsupervised time series representation learning.

## Quick Start
```bash
# Training TimesURL
python train.py --dataset Chinatown --batch-size 8 --epochs 200 --loader UCR
```

## Files Structure
- `timesurl.py`: Main model implementation
- `train.py`: Training script

## Usage
This directory contains the core TimesURL implementation. Use this for baseline comparisons or when you need the original TimesURL functionality without temperature scheduling extensions.

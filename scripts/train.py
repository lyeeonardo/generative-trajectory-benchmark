#!/usr/bin/env python
"""Train a joint model; smoke testing is the default, full training is explicit."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from generators.registry import available_models
from training.train_joint import train

def validate_config(config):
    if config['model']['method'] not in available_models():
        raise ValueError('Unknown registered joint model')
    tr=config['training']
    if not 0<tr['batch_size'] or not 0<tr['initial_windows']<=tr['maximum_windows']:
        raise ValueError('Positive batch size and ordered training budgets are required')
    if any(tr[k]<=0 for k in ('learning_rate','validation_windows','checkpoint_windows','progress_seconds')):
        raise ValueError('Positive learning rate and reporting intervals are required')
    return config

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--stage',choices=['smoke','development','pilot','full'],default='smoke')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke-steps',type=int,default=8)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--target-windows',type=int)
    args=parser.parse_args()
    if args.smoke_steps<1:parser.error('--smoke-steps must be positive')
    config=validate_config(json.loads(args.config.read_text()))
    from scripts.study_guard import require_training_ready
    require_training_ready(config,args.stage)
    output=args.output or ROOT/'outputs/training'/config['fit_id']
    return train(config,output,stage=args.stage,resume=args.resume,smoke_steps=args.smoke_steps,device=args.device,target_windows=args.target_windows)

if __name__=='__main__':
    main()

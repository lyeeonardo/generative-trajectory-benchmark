#!/usr/bin/env python
"""Audit, prepare, or regenerate the active offline dataset."""
from pathlib import Path
import argparse, json, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.cache import build, audit

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["audit","prepare","generate"])
    p.add_argument("--dataset",type=Path,default=ROOT/"datasets/training")
    p.add_argument("--output",type=Path)
    p.add_argument("--workers",type=int,default=12)
    p.add_argument("--max-cases",type=int)
    args=p.parse_args()
    if args.command=="generate":
        if args.output is None:p.error("generate requires --output for a fresh dataset")
        from data.generation import generate
        print(json.dumps(generate(args.output,args.dataset,args.workers,args.max_cases),indent=2))
    elif args.command=="prepare":
        print(build(args.dataset))
        audit(args.dataset)
    else:audit(args.dataset)
if __name__=="__main__":main()

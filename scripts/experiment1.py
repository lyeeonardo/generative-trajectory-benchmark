#!/usr/bin/env python
"""Evaluate one joint model: oracle K1 behavior, capability panels, or scaling."""
from pathlib import Path
import argparse, json, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.common import writable_output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--panel",choices=["behavior","capabilities","scaling"],default="behavior")
    p.add_argument("--split",choices=["development","test"],default="development")
    p.add_argument("--policy",choices=["every_step","agreement","both"],default="both")
    p.add_argument("--cases",type=Path)
    p.add_argument("--calibration",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--max-cases",type=int)
    p.add_argument("--max-steps",type=int)
    p.add_argument("--sampling-temperature",type=float,default=1.)
    p.add_argument("--guidance-scale",type=float,default=1.)
    p.add_argument("--tilt-label",choices=["oracle","wrong"],default="oracle",help="Controlled supplied-tilt intervention for behavior only")
    p.add_argument("--smoke",action="store_true",help="Small scaling panel on development snapshots")
    a=p.parse_args();output=writable_output(a.output)
    contract=json.loads((ROOT/'configs/experiment1.json').read_text())
    if a.split=='test':
        from scripts.study_guard import require_ready
        require_ready(contract)
    elif not contract.get('development_execution_ready',False):
        raise RuntimeError('Development evaluator pilot is not ready')
    if a.panel!="behavior" and (a.max_cases is not None or a.max_steps is not None):p.error("--max-cases/--max-steps apply only to behavior")
    if a.panel!="behavior" and a.tilt_label!="oracle":p.error("--tilt-label applies only to behavior")
    if a.panel=="behavior":
        from evaluation.experiment1 import evaluate
        cases=a.cases or ROOT/"datasets/evaluation"/("test_cases.json" if a.split=="test" else "development_cases.json")
        supplied=None
        if a.tilt_label=="wrong":
            from evaluation.tilt_conditioning import label_schedule,supplied_labels
            supplied=supplied_labels(label_schedule(json.loads(cases.read_text())),"wrong")
        policies=["every_step","agreement"] if a.policy=="both" else [a.policy]
        records={policy:evaluate(a.checkpoint,cases,output/policy,policy,a.device,a.max_cases,a.max_steps,
            supplied_tilts=supplied,sampling_temperature=a.sampling_temperature,guidance_scale=a.guidance_scale) for policy in policies}
        print(json.dumps({k:{x:v[x] for x in ("status","successes","started_episodes","cached_step_fraction","model_checker_seconds_per_action") if x in v} for k,v in records.items()},indent=2))
    elif a.panel=="capabilities":
        from evaluation.capabilities import capabilities
        report=capabilities(a.checkpoint,a.split,output,calibration=a.calibration,device=a.device,sampling_temperature=a.sampling_temperature,guidance_scale=a.guidance_scale)
        print(json.dumps({k:report[k] for k in ["status","snapshots","prediction_qualified","method"]},indent=2))
    else:
        from evaluation.scaling import evaluate
        if a.split=="development" and not a.smoke:p.error("development scaling requires --smoke; full scaling uses --split test")
        report=evaluate(a.checkpoint,output,device=a.device,smoke=a.smoke,sampling_temperature=a.sampling_temperature,guidance_scale=a.guidance_scale)
        print(json.dumps({k:report[k] for k in ["status","query_bundles","raw_sequences","method"]},indent=2))
if __name__=="__main__":main()

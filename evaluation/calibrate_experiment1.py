"""Fit one selected E1 checkpoint's observation likelihood on CAL only."""
from __future__ import annotations
import json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from aif.observation import calibrate
from data.banks import load_bank
from data.preparation import sha,write_json
from evaluation.capabilities import fixed_truth,numpy_dict
from evaluation.common import PUBLIC,CONFIG,atomic_npz,configure,identity_file,load_model,source_identity


def fit(checkpoint,output,device='cuda',samples=32,sampling_temperature=1.,guidance_scale=1.):
    checkpoint=Path(checkpoint);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if samples<2:raise ValueError('Calibration needs at least two predictive samples')
    identity={'checkpoint_sha256':sha(checkpoint),'split':'calibration','samples':samples,
              'experiment1_full_sha256':sha(CONFIG/'experiment1_full.json'),'sampling_temperature':sampling_temperature,'guidance_scale':guidance_scale,
              'bank_manifest_sha256':sha(Path('datasets/evaluation/manifest.json')),
              'sources':source_identity(),'source_sha256':sha(Path(__file__))}
    identity_file(output/'contract.json',identity)
    receipt=output/'calibration.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['checkpoint_sha256']!=identity['checkpoint_sha256']:raise ValueError('Calibration checkpoint changed')
        return saved
    started=time.perf_counter();configure(401,'cuda' if device.startswith('cuda') else device)
    model,_=load_model(checkpoint,device)
    if not np.isfinite(sampling_temperature) or sampling_temperature<=0:raise ValueError('Sampling temperature must be finite and positive')
    if not np.isfinite(guidance_scale) or guidance_scale<1:raise ValueError('Guidance scale must be finite and at least one')
    if model.config.method!='diffusion' and guidance_scale!=1.:raise ValueError('Guidance override is diffusion-only')
    model.sampling_temperature=float(sampling_temperature);model.guidance_scale=float(guidance_scale)
    bank,meta=load_bank('calibration');n=len(meta)
    context={k:np.repeat(bank[k],16,axis=0) for k in PUBLIC}
    actions=bank['actions'].reshape(-1,6,3)[:,:1];prediction_path=output/'predictions.npz'
    if prediction_path.exists():
        with np.load(prediction_path) as f:predictions={k:f[k] for k in f.files}
    else:
        rng=torch.Generator(device=device).manual_seed(9401);parts=[]
        for start in range(0,len(actions),32):
            parts.append(numpy_dict(model.predict({k:v[start:start+32] for k,v in context.items()},actions[start:start+32],samples=samples,generator=rng)))
        predictions={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        atomic_npz(prediction_path,**predictions)
    if predictions['observations'].shape!=(n*16,samples,1,7):raise ValueError('Unexpected calibration prediction shape')
    # Truth is deliberately opened only after every prediction is committed.
    truth=fixed_truth(bank,meta,prediction_path);target=truth['observations'].reshape(-1,6,7)[:,0]
    valid=truth['valid'].reshape(-1,6)[:,0]
    if not valid.all():
        predictive=predictions['observations'][valid,:,0];target=target[valid]
    else:predictive=predictions['observations'][:,:,0]
    extra=calibrate(torch.as_tensor(predictive),torch.as_tensor(target)).tolist()
    result={'status':'COMPLETE','split':'calibration','checkpoint_sha256':sha(checkpoint),
            'snapshots':n,'fixed_action_branches':n*16,'valid_rows':int(valid.sum()),
            'prediction_samples':samples,'sampling_temperature':sampling_temperature,'guidance_scale':guidance_scale,'extra_variance':extra,'prediction_sha256':sha(prediction_path),
            'elapsed_seconds':time.perf_counter()-started,'test_accessed':False,
            'scope':'CAL-only likelihood width fit after development checkpoint selection; no recipe or checkpoint tuning'}
    write_json(receipt,result);return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--samples',type=int,default=32)
    parser.add_argument('--sampling-temperature',type=float,default=1.)
    parser.add_argument('--guidance-scale',type=float,default=1.)
    args=parser.parse_args()
    print(json.dumps(fit(args.checkpoint,args.output,args.device,args.samples,args.sampling_temperature,args.guidance_scale),indent=2))


if __name__=='__main__':main()

"""Calibrated observation moments and likelihood for offline model evaluation.

Six independent Gaussian physical coordinates and a normalized wrapped Gaussian
for yaw. This is a calibrated sample-moment approximation, not a native model
likelihood. Online filtering and MI use the NumPy law in aif.likelihood.
No physics engine or privileged state is available here.
"""
from __future__ import annotations
import math
import torch

FLOOR_STD=(.002,.002,.02,.02,.002,.002,.02)

def moments(samples):
    mean=samples.mean(-2);mean=mean.clone()
    mean[...,6]=torch.atan2(samples[...,6].sin().mean(-1),samples[...,6].cos().mean(-1))
    delta=samples-mean.unsqueeze(-2);delta=delta.clone();delta[...,6]=torch.atan2(delta[...,6].sin(),delta[...,6].cos())
    return mean,delta.square().mean(-2)

class ObservationLaw:
    def __init__(self,mean,variance):
        if mean.shape!=variance.shape or mean.shape[-1]!=7:raise ValueError('Expected (...,7) moments')
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all() or (variance<=0).any():raise ValueError('Invalid observation moments')
        self.mean=mean;self.variance=variance.clone();self.variance[...,6].clamp_(max=math.pi**2)
    def log_prob(self,y):
        d=y-self.mean;v=self.variance
        normal=(-.5*(d[...,:6].square()/v[...,:6]+v[...,:6].log()+math.log(2*math.pi))).sum(-1)
        angle=torch.atan2(d[...,6].sin(),d[...,6].cos())
        shifts=torch.arange(-8,9,device=y.device,dtype=y.dtype)*2*math.pi
        wrapped=torch.logsumexp(-.5*((angle[...,None]+shifts).square()/v[...,6,None]+v[...,6,None].log()+math.log(2*math.pi)),-1)
        return normal+wrapped
    def sample(self, sample_shape=(),generator=None):
        eps=torch.randn(tuple(sample_shape)+tuple(self.mean.shape),device=self.mean.device,generator=generator)
        y=self.mean+eps*self.variance.sqrt();y[...,6]=torch.atan2(y[...,6].sin(),y[...,6].cos());return y

def law_from_samples(samples, extra_variance):
    mean,var=moments(samples);extra=torch.as_tensor(extra_variance,device=samples.device,dtype=samples.dtype)
    return ObservationLaw(mean,var+extra)

def calibrate(samples,truth):
    mean,var=moments(samples);res=truth-mean;res=res.clone();res[...,6]=torch.atan2(res[...,6].sin(),res[...,6].cos())
    floor=torch.tensor(FLOOR_STD,device=truth.device).square()
    extra=torch.maximum((res.square()-var).mean(0),floor)
    return extra

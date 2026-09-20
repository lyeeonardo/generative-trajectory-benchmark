"""Shared public context, checkpoint loading, runtime setup and proposal sampling."""
from pathlib import Path
from dataclasses import replace, fields
import copy, hashlib, json, math, os
import numpy as np
import torch
from data.preparation import ROOT, sha, write_json
from data.constants import PUBLIC
from aif.operational_cost import horizon_cost, DEFAULT_WEIGHTS
from data.joint_training import encode_batch
from generators.joint_world_model import JointConfig
from generators.registry import create_model, DISPLAY_NAMES
from training.train_joint import setup
CONFIG = ROOT / "configs"
BASE = ROOT / "outputs/experiment1"
BANK = ROOT / "datasets/evaluation"
METHODS = DISPLAY_NAMES
def atomic_npz(path, **arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def public_context(history, past, geometry, tilt, step, limit=100):
    ho=np.zeros((1,5,7),np.float32);ha=np.zeros((1,4,3),np.float32);hm=np.zeros((1,5),bool)
    n=min(len(history),5);ho[0,-n:]=history[-n:];hm[0,-n:]=True
    n=min(len(past),4)
    if n:ha[0,-n:]=past[-n:]
    return dict(history_observations=ho,history_actions=ha,history_mask=hm,geometry=np.asarray(geometry,np.float32)[None],
                tilt=np.asarray(tilt,np.float32)[None],time_fraction=np.asarray([step/limit],np.float32))


def forbid_hypothetical(*args,**kwargs):
    raise RuntimeError('Hypothetical simulator access is forbidden in Experiment 1 K1 execution')


def environment(collection,scene):
    from environment.task import UphillTask
    from mujoco_task.sim.scene import SceneSpec
    env=UphillTask(max_steps=int(collection['simulator']['max_steps']))
    obs=env.reset(SceneSpec(**scene))
    env.copy=forbid_hypothetical
    env.set_state=forbid_hypothetical
    env.restore=forbid_hypothetical
    return env,obs


def shadow_environment(collection,scene):
    # Compatibility import for old callers; active execution always stops at success.
    return environment(collection,scene)


def episode_seed(case_id,evaluation_seed):
    # Same schedule across methods/training repeats; no architecture-based seed choice.
    return int.from_bytes(hashlib.sha256(f'e1:{case_id}:{evaluation_seed}'.encode()).digest()[:8],'little') % (2**63-1)


def actual_cost(history,past,events,geometry):
    if past:
        return horizon_cost(np.asarray(history),np.asarray(past),np.asarray(events),np.ones(len(past),bool),geometry,np.zeros(3))
    # Zero executed transitions after an invalid first model output: no synthetic step.
    parts=np.zeros(len(DEFAULT_WEIGHTS),np.float64)
    parts[0]=DEFAULT_WEIGHTS['goal_terminal']*np.linalg.norm(np.asarray(history[-1])[:2]-geometry[:2])
    return float(parts.sum()),parts

def configure(seed,device):
    setup(seed,device)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False)
    if device.startswith('cuda'):
        import subprocess
        power=subprocess.check_output(['nvidia-smi','--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True)
        if any(abs(float(p)-450)>1 for p in power.splitlines()):raise RuntimeError('450 W GPU power limit required')

def identity_file(path,record):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=record:raise ValueError('Frozen identity changed: '+str(path))
    else:write_json(path,record)


@torch.no_grad()
def propose_independent(model,context,generators):
    """Same native sampler, one independent RNG per episode, dynamic batched calls."""
    rows=[]
    for i,generator in enumerate(generators):
        row=model.propose_joint({k:v[i:i+1] for k,v in context.items()},K=1,H=6,quality=2,generator=generator)
        rows.append({k:v[:,0] for k,v in row.items()})
    return {k:torch.cat([row[k] for row in rows],dim=0) for k in rows[0]}


def selection_key(r):
    return (-r['successes'],round(r['mean_actual_cost'],8),round(r['one_step_nll'],8),round(r['median_K1_generation_seconds'],8),r['method'])


def choose_method(rows):
    eligible=[r for r in rows if r['prediction_qualified'] and r['interface_qualified']]
    return min(eligible,key=selection_key)['method'] if eligible else None


def load_model(checkpoint, device="cuda"):
    """Load trusted local weights; verify imported artifact bytes before deserialization.

    Imported checkpoint payloads retain their original provenance. The active
    manifest supplies the canonical model name without rewriting frozen weights.
    New checkpoints use canonical names directly.
    """
    path = Path(checkpoint).resolve()
    manifest_path = ROOT / "checkpoints/manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entries = manifest.get("checkpoints", [])
    if isinstance(entries, dict): entries = list(entries.values())
    entry = next((x for x in entries if (ROOT/x["path"]).resolve() == path), None)
    if path.is_relative_to((ROOT/"checkpoints").resolve()) and entry is None:
        raise ValueError("Checkpoint is absent from the import manifest")
    if entry is not None and sha(path) != entry["sha256"]:
        raise ValueError("Imported checkpoint hash mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(saved.get("model_config", saved.get("contract", {}).get("config", {}).get("model", {})))
    if entry is not None: cfg["method"] = entry["method"]
    if entry is not None and cfg['method'] in ('diffusion','flow_matching','autoregressive'):
        cfg = {key: value for key,value in cfg.items() if key in {f.name for f in fields(JointConfig)}}
    norm = saved.get("normalization")
    if norm is None: raise ValueError("Checkpoint must embed its training normalization")
    model = create_model(cfg, norm).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    saved = dict(saved)
    saved["model_config"] = cfg
    saved["contract"] = copy.deepcopy(saved.get("contract", {}))
    saved["contract"].setdefault("config", {})["model"] = cfg
    if entry is not None: saved["contract"]["config"]["seed"] = entry["seed"]
    return model, saved


def writable_output(output):
    path = Path(output).resolve()
    for name in ("archive", "datasets", "checkpoints", "results"):
        if path.is_relative_to((ROOT/name).resolve()):
            raise ValueError("New runs must use an output directory outside preserved artifacts")
    return path


def source_identity():
    """Bind resumable evaluation to all active inference and physical dependencies."""
    folders=("aif","data","environment","envs","evaluation","generators","mujoco_task","training")
    paths=sorted(p for folder in folders for p in (ROOT/folder).rglob("*.py"))
    return {str(path.relative_to(ROOT)):sha(path) for path in paths}

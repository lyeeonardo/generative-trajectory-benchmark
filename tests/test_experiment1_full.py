import json
from pathlib import Path
from scripts import experiment1_full as full

ROOT=Path(__file__).resolve().parents[1]

def test_full_contract_includes_every_family_and_three_fits_without_qualification_gate():
    c=json.loads((ROOT/'configs/experiment1_full.json').read_text())
    assert c['families']==['diffusion','autoregressive','cvae','flow_matching']
    assert c['seeds']==[13] and c['fits']==4
    assert 'regardless' in c['qualification_policy']
    assert c['weak_model_policy'].startswith('No architecture')
    assert c['splits']['checkpoint_selection']=='development only'
    assert c['splits']['final_reporting'].startswith('one sealed test campaign')
    cap=c['training_budget_policy']['maximum_sampled_windows_per_fit']
    assert cap==c['recipes']['diffusion']['training']['budget_windows']==53_705_000
    assert all(c['recipes'][family]['training']['budget_windows']==cap for family in c['families'])
    assert c['checkpoint_selection']['maximum_windows_inclusive']==cap


def test_every_full_recipe_uses_frozen_data_oracle_tilt_and_substantial_budget():
    train_rows=53705
    for family,seed in full.fits():
        r=full.recipe(family,seed)
        assert r['model']['method']==family
        assert r['seed']==seed and r['training_membership']=='all'
        assert r['oracle_test_guidance']==['true_tilt','HIGH_request']
        assert r['full_training_ready'] is True
        assert r['training']['initial_windows']==r['training']['maximum_windows']
        assert r['training']['initial_windows']==1000*train_rows
        assert r['supervision']['high_success_demo_fraction']==1.0


def test_development_rank_prefers_qualified_then_useful_then_likelihood_and_error():
    base=dict(prediction_qualified=True,raw_action_usefulness=.8,one_step_nll=-2.,fixed_prediction_rmse=.3,windows=100)
    assert full.metric_rank({**base,'prediction_qualified':True})>full.metric_rank({**base,'prediction_qualified':False,'raw_action_usefulness':1.})
    assert full.metric_rank({**base,'raw_action_usefulness':.9})>full.metric_rank(base)
    assert full.metric_rank({**base,'one_step_nll':-3.})>full.metric_rank(base)
    assert full.metric_rank({**base,'fixed_prediction_rmse':.2})>full.metric_rank(base)


def test_test_panels_keep_one_candidate_control_and_complete_raw_pool():
    c=json.loads((ROOT/'configs/experiment1_full.json').read_text())
    e=c['evaluation']
    assert e['behavior_episodes_per_fit_per_arm']==90
    assert e['behavior_arms']==['every_step','agreement_reuse','wrong_tilt_every_step']
    assert c['common']['primary_behavior_K']==1
    assert e['raw_proposals_per_snapshot_per_quality']==32
    assert e['test_repetition'].startswith('One sealed campaign')
    protocol=json.loads((ROOT/'datasets/evaluation/test_cases.json').read_text())
    assert len(protocol['cases'])==9
    assert protocol['evaluation_rng_seeds']==list(range(101,111))
    assert protocol['episodes_per_fit']==90
    assert 'proposal-trial seed' in e['uncertainty']['paired_behavior']
    assert 'retraining-variation' in e['uncertainty']['training_scope']


def test_checkpoint_eligibility_excludes_any_over_budget_evidence(tmp_path):
    budget=53_705_000
    history=[{'windows':10_741_000},{'windows':53_705_000},{'windows':53_706_752}]
    folder=tmp_path/'checkpoints';folder.mkdir()
    for row in history:
        (folder/f"w{row['windows']:012d}.pt").touch()
    candidates,cap=full.eligible_candidates(history,tmp_path,budget)
    assert cap==budget
    assert [row['windows'] for row in candidates]==[10_741_000,53_705_000]


def test_calibration_entrypoint_resolves_repo_imports():
    import subprocess,sys
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/calibrate_experiment1.py'),'--help'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr

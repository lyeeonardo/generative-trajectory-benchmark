"""Stage 1 diagnostic plots and simple rollout videos."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def plot_topk_rollout_overlay(plan_result, path: str | Path, *, title: str = "Top-K Rollouts") -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 6))
    obs0 = plan_result.rollouts[0].observations[0]
    obstacle = plt.Circle(obs0[9:11], float(obs0[11]), color="black", alpha=0.25)
    goal = plt.Circle(obs0[7:9], 0.12, color="green", alpha=0.25)
    ax.add_patch(obstacle)
    ax.add_patch(goal)
    probs = np.asarray(plan_result.policy_posterior)
    top = set(np.argsort(probs)[-min(5, len(probs)) :].tolist())
    for index, rollout in enumerate(plan_result.rollouts):
        xy = rollout.observations[:, :2]
        if index == plan_result.selected_index:
            ax.plot(xy[:, 0], xy[:, 1], color="red", linewidth=2.5, label="selected")
        elif rollout.collision:
            ax.plot(xy[:, 0], xy[:, 1], color="black", alpha=0.25, linewidth=0.8)
        elif index in top:
            ax.plot(xy[:, 0], xy[:, 1], color="blue", alpha=0.65, linewidth=1.4)
        else:
            ax.plot(xy[:, 0], xy[:, 1], color="gray", alpha=0.18, linewidth=0.8)
    ax.scatter([obs0[0]], [obs0[1]], c="orange", s=35, label="start")
    ax.set_title(title)
    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


def plot_energy_over_time(step_logs: list[dict[str, float]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    keys = ["G_total", "G_preference", "G_safety", "G_smoothness", "G_control"]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(step_logs))
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in step_logs]
        ax.plot(xs, values, label=key)
    ax.set_xlabel("decision")
    ax.set_ylabel("energy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


def render_episode_gif(trajectory: np.ndarray, scene_obs: np.ndarray, path: str | Path, *, fps: int = 8) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    trajectory = np.asarray(trajectory, dtype=np.float32)
    for end in range(1, len(trajectory) + 1):
        fig, ax = plt.subplots(figsize=(4, 5))
        obstacle = plt.Circle(scene_obs[9:11], float(scene_obs[11]), color="black", alpha=0.25)
        goal = plt.Circle(scene_obs[7:9], 0.12, color="green", alpha=0.25)
        ball = plt.Circle(trajectory[end - 1], 0.04, color="red", alpha=0.9)
        ax.add_patch(obstacle)
        ax.add_patch(goal)
        ax.plot(trajectory[:end, 0], trajectory[:end, 1], color="red", linewidth=1.5)
        ax.add_patch(ball)
        ax.set_xlim(-0.85, 0.85)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frame = rgba[:, :, :3].copy()
        frames.append(frame)
        plt.close(fig)
    imageio.mimsave(destination, frames, duration=1000 / max(int(fps), 1))



def plot_policy_posterior_over_time(posterior_logs: list[dict[str, float]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    xs = np.arange(len(posterior_logs))
    fig, ax1 = plt.subplots(figsize=(7, 4))
    entropy = [float(row.get("posterior_entropy", 0.0)) for row in posterior_logs]
    max_prob = [float(row.get("max_probability", 0.0)) for row in posterior_logs]
    rank = [float(row.get("selected_candidate_rank_by_G", 0.0)) for row in posterior_logs]
    ax1.plot(xs, entropy, label="entropy")
    ax1.plot(xs, max_prob, label="max probability")
    ax1.set_xlabel("decision")
    ax1.set_ylabel("posterior")
    ax2 = ax1.twinx()
    ax2.plot(xs, rank, color="tab:red", alpha=0.5, label="selected rank")
    ax2.set_ylabel("rank")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


def plot_route_distribution_over_time(route_logs: list[dict[str, float]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    xs = np.arange(len(route_logs))
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ("proposal_left", "proposal_right", "proposal_invalid", "posterior_left", "posterior_right"):
        ax.plot(xs, [float(row.get(key, 0.0)) for row in route_logs], label=key)
    ax.set_xlabel("decision")
    ax.set_ylabel("fraction / mass")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


def plot_runtime_decomposition(runtime_logs: list[dict[str, float]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    xs = np.arange(len(runtime_logs))
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ("proposal_time", "rollout_scoring_time", "planning_time"):
        ax.plot(xs, [float(row.get(key, 0.0)) for row in runtime_logs], label=key)
    ax.set_xlabel("decision")
    ax.set_ylabel("seconds")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


def plot_proposal_cloud_comparison(plans_by_model: dict[str, object], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = max(len(plans_by_model), 1)
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 5), squeeze=False)
    for ax, (model, plan) in zip(axes[0], plans_by_model.items()):
        obs0 = plan.rollouts[0].observations[0]
        ax.add_patch(plt.Circle(obs0[9:11], float(obs0[11]), color="black", alpha=0.20))
        ax.add_patch(plt.Circle(obs0[7:9], 0.12, color="green", alpha=0.20))
        for rollout in plan.rollouts:
            xy = rollout.observations[:, :2]
            color = {"left": "tab:blue", "right": "tab:orange", "center": "tab:green"}.get(rollout.route_label, "black")
            ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=0.35, linewidth=0.9)
            ax.scatter([xy[-1, 0]], [xy[-1, 1]], color=color, s=10, alpha=0.55)
        ax.scatter([obs0[0]], [obs0[1]], c="red", s=25)
        ax.set_title(model)
        ax.set_xlim(-0.85, 0.85)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


def plot_posterior_selected_comparison(plans_by_model: dict[str, object], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    first_plan = next(iter(plans_by_model.values()))
    obs0 = first_plan.rollouts[0].observations[0]
    ax.add_patch(plt.Circle(obs0[9:11], float(obs0[11]), color="black", alpha=0.20))
    ax.add_patch(plt.Circle(obs0[7:9], 0.12, color="green", alpha=0.20))
    for model, plan in plans_by_model.items():
        rollout = plan.rollouts[plan.selected_index]
        xy = rollout.observations[:, :2]
        selected_score = plan.score_breakdowns[plan.selected_index].G_total
        ax.plot(xy[:, 0], xy[:, 1], linewidth=2.0, label=f"{model} G={selected_score:.2f}")
    ax.scatter([obs0[0]], [obs0[1]], c="red", s=30, label="start")
    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


def plot_runtime_pareto(aggregate: dict[str, object], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, payload in aggregate.get("models", {}).items():
        summary = payload.get("summary", {})
        x = float(summary.get("planning_time_per_decision", 0.0))
        y = float(summary.get("success_rate", 0.0))
        ax.scatter([x], [y], s=45)
        ax.annotate(model, (x, y), fontsize=8)
    ax.set_xlabel("planning time / decision (s)")
    ax.set_ylabel("success rate")
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


def plot_diversity_quality(aggregate: dict[str, object], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, payload in aggregate.get("models", {}).items():
        summary = payload.get("summary", {})
        x = float(summary.get("mean_route_entropy", 0.0))
        y = float(summary.get("mean_best_of_K_G_total", summary.get("mean_metric_best_of_K_G_total", 0.0)))
        ax.scatter([x], [y], s=45)
        ax.annotate(model, (x, y), fontsize=8)
    ax.set_xlabel("route entropy")
    ax.set_ylabel("best-of-K G_total")
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)

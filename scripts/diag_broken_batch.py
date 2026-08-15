#!/usr/bin/env python
"""Diagnose the fixed huge-loss batches in pi05 membench training.

Loads a trained checkpoint, streams batches through the real training data
pipeline, and the moment a batch's mean loss is huge (>1000), dumps the
per-sample / per-position loss structure plus v_t / u_t / time / input stats
to reveal which tokens produce the fixed garbage activations.

Usage:
    CUDA_VISIBLE_DEVICES=<gpu> python scripts/diag_broken_batch.py \
        --config pi05_membench_wr04_lora \
        --checkpoint /path/to/ckpt/5000 \
        [--batch-size 8] [--max-batches 200]
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src",):
    if str(source) not in os.sys.path:
        os.sys.path.insert(0, str(source))

# Match the training pipeline: only decode the two cameras the policy consumes.
os.environ.setdefault(
    "OPENPI_MEMBENCH_CAMERA_KEYS",
    "observation.images.robot0_agentview_right,observation.images.robot0_eye_in_hand",
)

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding


def compute_loss_detailed(model, rng, observation, actions):
    """Mirror pi0.Pi0Model.compute_loss but keep the intermediates for inspection."""
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=True)

    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])
    per_pos_loss = jnp.square(v_t - u_t).mean(axis=-1)  # [b, horizon]

    return {
        "per_pos_loss": per_pos_loss,  # [b, horizon]
        "v_t": v_t,
        "u_t": u_t,
        "time": time,
        "noise": noise,
        "x_t": x_t,
        "input_mask": input_mask,
        "prefix_mask": prefix_mask,
    }


def dump_batch(obs, actions, r, batch_idx):
    b, h = r["per_pos_loss"].shape
    per_sample = np.array(r["per_pos_loss"]).mean(axis=1)  # [b]
    per_pos = np.array(r["per_pos_loss"]).mean(axis=0)     # [h]
    v = np.asarray(r["v_t"], dtype=np.float64)
    u = np.asarray(r["u_t"], dtype=np.float64)
    x = np.asarray(r["x_t"], dtype=np.float64)
    noise = np.asarray(r["noise"], dtype=np.float64)
    time = np.asarray(r["time"], dtype=np.float64)

    print(f"  batch={batch_idx} shape=({b},{h})")
    print(f"  per-sample loss: min={float(per_sample.min()):.2f} max={float(per_sample.max()):.2f}")
    print(f"    每样本均值: {[round(float(xv),1) for xv in per_sample[:b]]}")
    print(f"  per-position loss: argmax_pos={int(per_pos.argmax())} "
          f"| 后5位: {[round(float(xv),1) for xv in per_pos[-5:]]}")
    print(f"  time: min={float(time.min()):.4f} max={float(time.max()):.4f}")
    print(f"  |v_t| max={float(np.abs(v).max()):.1f} | |u_t| max={float(np.abs(u).max()):.1f} "
          f"| |x_t| max={float(np.abs(x).max()):.1f} | |noise| max={float(np.abs(noise).max()):.1f}")

    # 定位坏样本
    bad_sample = int(per_sample.argmax())
    print(f"  >>> 坏样本 index={bad_sample}")
    print(f"  >>> 该样本 time={float(time[bad_sample]):.4f}")
    print(f"  >>> 该样本 actions[48]（第49帧）: {[round(float(xv),4) for xv in actions[bad_sample,48,:]]}")
    print(f"  >>> 该样本 actions[49]（第50帧/最后一帧）: {[round(float(xv),4) for xv in actions[bad_sample,49,:]]}")
    print(f"  >>> 该样本 actions[49] 与 [48] 是否完全相同: {bool(np.array_equal(np.asarray(actions[bad_sample,49,:]), np.asarray(actions[bad_sample,48,:]))) if actions[bad_sample,49].ndim else 'n/a'}")
    print(f"  >>> 该样本 u_t[49]（目标噪声）前8维: {[round(float(xv),3) for xv in u[bad_sample,49,:8]]}")
    print(f"  >>> 该样本 v_t[49]（模型预测）前8维: {[round(float(xv),1) for xv in v[bad_sample,49,:8]]}")
    print(f"  >>> 该样本 x_t[49]（含噪动作）前8维: {[round(float(xv),3) for xv in x[bad_sample,49,:8]]}")
    # 全 batch action[49] 每样本的值（看是否只有坏样本异常）
    print(f"  全batch actions[:,49,0:3]:")
    for s in range(b):
        print(f"    sample {s}: {[round(float(xv),4) for xv in actions[s,49,0:3]]}")
    # 该样本在第48/49帧是否为 episode 末尾 clamp（对比更早帧）
    print(f"  坏样本 actions[0] vs [49] 前3维: {[round(float(xv),4) for xv in actions[bad_sample,0,:3]]} vs {[round(float(xv),4) for xv in actions[bad_sample,49,:3]]}")

    # inputs
    imgs = {k: np.asarray(vv) for k, vv in obs.images.items() if k in obs.image_masks and obs.image_masks[k]}
    for k, img in imgs.items():
        print(f"  img[{k}] shape={img.shape} min={float(img.min()):.2f} max={float(img.max()):.2f} mean={float(img.mean()):.2f}")
    state = np.asarray(obs.state)
    act = np.asarray(actions)
    print(f"  state shape={state.shape} min={float(state.min()):.3f} max={float(state.max()):.3f} nan={int(np.isnan(state).sum())}")
    print(f"  actions shape={act.shape} min={float(act.min()):.3f} max={float(act.max()):.3f} nan={int(np.isnan(act).sum())}")
    nz = (np.abs(act) > 1e-6).mean(axis=(0, 1))
    print(f"  actions 每维非零比例: {[round(float(xv),2) for xv in nz]}")
    pmask = np.asarray(obs.tokenized_prompt_mask)
    print(f"  tokenized_prompt: 有效token数(min/max)={int(pmask.sum(axis=1).min())}/{int(pmask.sum(axis=1).max())}")
    imask = np.asarray(r["input_mask"])
    print(f"  input_mask: 每样本 False 位置数 min={int((~imask).sum(axis=1).min())} max={int((~imask).sum(axis=1).max())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    jax.config.update("jax_compilation_cache_dir", str(ROOT / ".jax_cache"))

    print(f"Loading config {args.config} ...", flush=True)
    cfg = _config.get_config(args.config)
    print(f"Loading model from {args.checkpoint} ...", flush=True)
    model = cfg.model.load(_model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16))
    model.train()
    print("Model loaded.", flush=True)

    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    dl_cfg = dataclasses.replace(cfg, batch_size=args.batch_size)
    dl = _data_loader.create_data_loader(
        dl_cfg,
        sharding=data_sharding,
        shuffle=True,
        num_batches=args.max_batches,
    )
    it = iter(dl)

    rng = jax.random.key(args.seed)
    found = 0
    for batch_idx in range(args.max_batches):
        obs, actions = next(it)
        step_rng = jax.random.fold_in(rng, batch_idx)
        r = compute_loss_detailed(model, step_rng, obs, actions)
        mean_loss = float(jnp.mean(r["per_pos_loss"]))
        if mean_loss > 1000:
            found += 1
            print(f"\n=== BROKEN batch {batch_idx}: mean_loss={mean_loss:.1f} ===", flush=True)
            dump_batch(obs, actions, r, batch_idx)
            if found >= 2:
                break
        elif batch_idx % 10 == 0:
            print(f"  step {batch_idx}: loss={mean_loss:.4f} (clean)", flush=True)
    print(f"\nDone. broken batches found: {found} / {batch_idx + 1}", flush=True)


if __name__ == "__main__":
    main()

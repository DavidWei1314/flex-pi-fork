Action-to-visual gradient curriculum
===================================

Question: does delaying action supervision of the visual trunk improve control?
The full FlexPi architecture, losses, data and forward visibility stay unchanged.
RGB, DINO and pointmap losses keep their original weights (beta = 1).

Four Hydra presets:
  action_gradient=joint     alpha = 1 throughout
  action_gradient=detached  alpha = 0 throughout
  action_gradient=ramp      0 for 10%, linear 0 -> 1 until 40%, then 1
  action_gradient=weak      alpha = 0.75 throughout

Example, inside an allocated GPU runtime with prepared RoboTwin data/weights:
  NUM_EPISODES_PER_TASK=100 bash scripts/train_flexpi_robotwin.sh \
    action_gradient=ramp seed=42 max_steps=YOUR_FIXED_UPDATE_BUDGET \
    output_dir=/site/YOUR_USER/runs/flexpi-gradient/ramp-seed42

Use the SAME positive max_steps, global batch, data subset, data seed, model
initialization, learning-rate schedule, loss weights, stream dropout, inference
settings and rollout seeds across conditions. Repeat with training seeds 42/43/44,
pairing each seed across all four conditions. Data subset selection must remain
fixed across seeds as well. Measure throughput/memory before sizing full jobs.
Use identical Wan/ActionDiT initial weights, not a policy already trained on the
target benchmark, to test the early-training hypothesis. Train the complete 6B
model; small dimensions below are correctness tests, not research experiments.
The training entry now seeds model construction before random heads/adapters are
created; previously seeding first occurred in the trainer, after construction.
Every rank uses the same initialization seed, then rank-specific sampling seeds.

Mechanism
---------
For each middle HBridge layer, split attention by QUERY owner. Both query groups
still normalize over the entire allowed key set. Only the visual K/V seen by
action queries use:
  routed = x.detach() + alpha * (x - x.detach())
This equals x in the forward pass; its backward derivative is alpha. The actual
implementation uses direct x/detach for endpoints. Visual queries keep the
original K/V graph, preserving world supervision. Outer HBridge layers stay
stream-local. All four presets use the same split-query implementation, including
alpha=1. Omitting the preset preserves the original attention path. Cached and
compiled inference paths are unchanged; no inference-time alpha is needed.

Scope: this routes the visual expert and upstream visual feature producers. It
does not detach the shared language/proprio conditioning used directly by the
action expert. It does not freeze the visual backbone, rescale the entire action
loss, or change attention visibility. Mutual visual/action visibility is rejected
because it would introduce additional gradient paths. The implementation requires
the explicit 2D/3D boolean masks already used by FlexPi training (SDPA path).

The raw gradient on independent action parameters is unchanged at fixed weights.
Global gradient clipping and Adam state can still change actual updates, which is
why identical losses do not imply identical optimizer steps. Keep these settings
fixed and track clipping; do not interpret all later differences as direct routing.

Schedule/resume
---------------
Progress = completed optimizer updates / the originally estimated total updates.
All microbatches in one accumulation window use the same alpha. Skipped updates do
not advance it. The training metric is action_to_visual_grad_scale. Full-state
checkpoints record the schedule AND horizon; changing either on resume raises an
error. Weight-only initialization starts a new schedule. The 0.75 control matches
the ramp's continuous mean coefficient, approximately its discrete mean; it does
not equalize actual gradient norms or optimizer-weighted gradient integrals.

Validation on Praxis
--------------------
From the repository root, dependency installation uses a CPU allocation:
  bash experiments/action_gradient/submit_validation.sh install --qos=annotate-100

After installation succeeds, neural tests use one GPU:
  bash experiments/action_gradient/submit_validation.sh test --gres=gpu:1

Use BOM (default) or pass --partition=uni; the job rejects other partitions.
The numerical algebra checks disable TF32, including the NGC environment
override. A separate check covers bf16. This affects tests, not training policy.

The submission helper captures just the validation inputs with praxis sbatch;
this avoids unrelated simulator submodule/archive attributes in a whole-repo
snapshot. Source and the base commit are recorded, including uncommitted edits.

The scripts create the separate uv environment /site/$USER/wam-envs/flex-pi/.venv,
stage source to /data/code/$USER, and write test reports to
/site/$USER/wam-envs/flex-pi/validation/<job_id>. Installation covers the base
training package; RoboTwin simulator assets/dependencies are a separate step.
No model computation runs on the login node. Tests use the actual Wan/ActionDiT
blocks and MoT at reduced width, including HBridge, per-sample masks, masked-out
streams, bf16 and checkpointing. They verify forward invariance, action-only and
combined-loss gradients, unchanged action gradients, and schedule resume rules.
They do not establish full-model training throughput or task success.

Research evaluation
-------------------
Primary: closed-loop task success, paired rollout seeds, at least three training
seeds per condition; report per-task results and between-training-seed variation.
Secondary: learning curves and held-out scene generalization with an explicit
split. A model trained on randomized scenes must not call those same conditions
unseen/OOD. Keep full-joint and action-only evaluation settings fixed and report
them separately. World prediction metrics are diagnostics, not control success.
First compare the four alpha schedules; only then study reducing beta, since that
is a separate intervention. Novelty is not a prerequisite for this experiment.

"""CUDA regressions for the actual MoT blocks and gradient-routing contract."""

from contextlib import nullcontext
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import pytest
import torch
import torch.utils.checkpoint

from flexpi.models.action_dit import ActionDiT
from flexpi.models.mot import MoT, scale_gradient
from flexpi.models.wan_video_dit import WanVideoDiT, precompute_freqs_cis
from flexpi.utils.action_gradient import ActionGradientSchedule


@pytest.fixture
def cuda(monkeypatch):
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0")
    if not torch.cuda.is_available():
        pytest.fail("Run these neural-network checks inside a Slurm GPU allocation")
    # The algebraic sum-of-gradients checks need full FP32 matmuls. TF32 can
    # round a combined upstream gradient differently from two separate passes.
    # The bf16 test below separately exercises production mixed precision.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert not torch.backends.cuda.matmul.allow_tf32
    yield torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


def make_case(device, checkpointing=False, hbridge=False, batched=False):
    torch.manual_seed(71)
    video = WanVideoDiT(
        hidden_dim=32, in_dim=4, ffn_dim=64, out_dim=4, text_dim=16,
        freq_dim=16, eps=1e-6, patch_size=(1, 2, 2), num_heads=2,
        attn_head_dim=8, num_layers=4, has_image_input=False,
        use_gradient_checkpointing=checkpointing,
    )
    action = ActionDiT(
        hidden_dim=16, action_dim=7, ffn_dim=32, text_dim=16,
        freq_dim=16, eps=1e-6, num_heads=2, attn_head_dim=8,
        num_layers=4, use_gradient_checkpointing=checkpointing,
    )
    model = MoT(
        {"video": video, "action": action},
        mot_checkpoint_mixed_attn=checkpointing,
        hbridge_enabled=hbridge,
    ).to(device).train()
    # Three visual streams (RGB, DINO, geometry), then action. Masks include
    # clean anchors, future tokens, and different visibility per sample.
    lengths = [3, 3, 3, 4]
    sv, sa = sum(lengths[:-1]), lengths[-1]
    mask = torch.ones(sv + sa, sv + sa, dtype=torch.bool, device=device)
    mask[:sv, sv:] = False
    for anchor in (0, 3, 6):
        mask[anchor, :sv] = False
        mask[anchor, [0, 3, 6]] = True
    if batched:
        mask = mask.expand(2, -1, -1).clone()
        mask[1, sv:, 4:6] = False
        # A dropped visual stream can have completely masked queries.
        mask[1, 6:9, :] = False
        mask[1, :, 6:9] = False
    freqs = precompute_freqs_cis(8, end=sv).to(device).unsqueeze(1)
    inputs = dict(
        embeds_all={"video": torch.randn(2, sv, 32, device=device),
                    "action": torch.randn(2, sa, 16, device=device)},
        attention_mask=mask,
        freqs_all={"video": freqs, "action": freqs[:sa]},
        context_all={"video": {"context": torch.randn(2, 2, 32, device=device)},
                     "action": {"context": torch.randn(2, 2, 16, device=device)}},
        t_mod_all={"video": torch.randn(2, 6, 32, device=device) * 0.1,
                   "action": torch.randn(2, 6, 16, device=device) * 0.1},
    )
    if hbridge:
        offsets = [0, 3, 6, 9]
        inputs["sub_stream_lens"] = lengths
        inputs["sub_stream_self_masks"] = [
            mask[..., off:off + size, off:off + size]
            for off, size in zip(offsets, lengths)
        ]
    return model, inputs


def run_backward(model, inputs, scale, loss_kind, autocast=False):
    model.zero_grad(set_to_none=True)
    model.action_gradient_scale = scale
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else nullcontext()
    with ctx:
        out = model(**inputs)
        world_loss = out["video"].float().square().mean()
        action_loss = out["action"].float().square().mean()
        loss = {"world": world_loss, "action": action_loss,
                "both": world_loss + action_loss}[loss_kind]
    loss.backward()
    grads = {
        name: torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
            for p in expert.parameters()
        ]).detach().clone()
        for name, expert in model.mixtures.items()
    }
    return {key: value.detach() for key, value in out.items()}, grads


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("hbridge", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_gradient_routing_contract(cuda, checkpointing, hbridge, batched):
    model, inputs = make_case(cuda, checkpointing, hbridge, batched)
    original_out, original_grad = run_backward(model, inputs, None, "both")
    reference_out, action_grad = run_backward(model, inputs, 1.0, "action")
    _, world_grad = run_backward(model, inputs, 1.0, "world")
    assert action_grad["video"].norm() > 1e-6
    assert world_grad["video"].norm() > 1e-6
    assert torch.count_nonzero(world_grad["action"]) == 0
    for scale in (0.0, 0.5, 0.75, 1.0):
        out, grad = run_backward(model, inputs, scale, "action")
        for name in out:
            torch.testing.assert_close(out[name], reference_out[name], rtol=0, atol=0)
            torch.testing.assert_close(out[name], original_out[name], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(grad["video"], scale * action_grad["video"], rtol=3e-5, atol=2e-6)
        torch.testing.assert_close(grad["action"], action_grad["action"], rtol=0, atol=0)
        _, combined = run_backward(model, inputs, scale, "both")
        torch.testing.assert_close(
            combined["video"], world_grad["video"] + scale * action_grad["video"],
            rtol=5e-5, atol=3e-6,
        )
        torch.testing.assert_close(combined["action"], action_grad["action"], rtol=2e-5, atol=2e-6)
        if scale == 1.0:
            for name in combined:
                torch.testing.assert_close(combined[name], original_grad[name], rtol=5e-5, atol=3e-6)


def test_bfloat16_and_checkpoint_recomputation(cuda):
    model, inputs = make_case(cuda, checkpointing=True, hbridge=True, batched=True)
    ref, ref_grad = run_backward(model, inputs, 1.0, "action", autocast=True)
    out, grad = run_backward(model, inputs, 0.5, "action", autocast=True)
    for name in out:
        torch.testing.assert_close(out[name], ref[name], rtol=0, atol=0)
        assert torch.isfinite(grad[name]).all()
    torch.testing.assert_close(grad["video"], ref_grad["video"] * 0.5, rtol=0.03, atol=3e-4)
    torch.testing.assert_close(grad["action"], ref_grad["action"], rtol=0, atol=0)
    # Changing the runtime coefficient after forward must not alter its backward.
    model.zero_grad(set_to_none=True)
    model.action_gradient_scale = 0.5
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**inputs)["action"].float().square().mean()
    model.action_gradient_scale = 0.0
    loss.backward()
    actual = torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
        for p in model.mixtures["video"].parameters()
    ])
    torch.testing.assert_close(actual, grad["video"], rtol=0, atol=0)


def test_rejects_reverse_visibility_and_preserves_inference(cuda):
    model, inputs = make_case(cuda)
    model.action_gradient_scale = 0.5
    inputs["attention_mask"][1, -1] = True
    with pytest.raises(ValueError, match="visual queries not to read actions"):
        model(**inputs)
    inputs["attention_mask"][1, -1] = False
    model.eval()
    with torch.no_grad():
        model.action_gradient_scale = None
        expected = model(**inputs)
        model.action_gradient_scale = 0.0
        actual = model(**inputs)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_scale_gradient_identity(cuda):
    x = torch.randn(17, device=cuda, requires_grad=True)
    for scale in (0.0, 0.5, 0.75, 1.0):
        y = scale_gradient(x, scale)
        torch.testing.assert_close(y, x, rtol=0, atol=0)
        if scale:
            grad, = torch.autograd.grad(y.sum(), x)
            torch.testing.assert_close(grad, torch.full_like(x, scale), rtol=0, atol=0)
        else:
            assert not y.requires_grad


def test_schedule_and_resume_contract():
    schedule = ActionGradientSchedule(1000, 0.0, 1.0, 0.1, 0.4)
    for step, expected in ((0, 0), (100, 0), (250, 0.5), (400, 1), (999, 1)):
        assert schedule.value(step) == pytest.approx(expected)
    resumed = ActionGradientSchedule(**schedule.state_dict())
    resumed.validate_resume(schedule.state_dict())
    assert resumed.value(250) == schedule.value(250)
    with pytest.raises(ValueError, match="differs"):
        ActionGradientSchedule(2000, 0.0, 1.0, 0.1, 0.4).validate_resume(schedule.state_dict())
    with pytest.raises(ValueError, match="differs"):
        schedule.validate_resume(None)
    with pytest.raises(ValueError):
        ActionGradientSchedule(1000, float("nan"))
    with pytest.raises(ValueError):
        ActionGradientSchedule(1000, ramp_start=0.4, ramp_end=0.1)


def test_hydra_presets():
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        legacy = compose(config_name="train", overrides=["task=robotwin_unified_flex_3cam_384_1e-4"])
        assert legacy.get("action_gradient") is None
        for preset, expected in (("joint", 1.0), ("detached", 0.0), ("ramp", 0.5), ("weak", 0.75)):
            cfg = compose(config_name="train", overrides=[
                "task=robotwin_unified_flex_3cam_384_1e-4", f"action_gradient={preset}",
            ])
            schedule = ActionGradientSchedule(total_steps=1000, **dict(cfg.action_gradient))
            assert schedule.value(250) == pytest.approx(expected)


@pytest.fixture
def compose_training_config(monkeypatch):
    """Compose real CLI overrides and expose them to the resume entry point."""
    hydra_state = HydraConfig.instance()
    monkeypatch.setattr(hydra_state, "cfg", hydra_state.cfg)
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")

    def compose_config(overrides):
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            full_cfg = compose(config_name="train", overrides=overrides,
                               return_hydra_config=True)
        hydra_state.set_config(full_cfg)
        return OmegaConf.masked_copy(full_cfg, [key for key in full_cfg if key != "hydra"])

    return compose_config


@pytest.mark.parametrize("group,leaf_overrides,expected_end", [
    ("ramp", [], 0.6),
    ("weak", [], 0.6),
    ("ramp", ["action_gradient.ramp_end=0.7"], 0.7),
])
def test_saved_config_keeps_gradient_group_and_applies_leaves(
    tmp_path, compose_training_config, group, leaf_overrides, expected_end,
):
    from flexpi.runtime import _maybe_swap_with_saved_config

    state_dir = tmp_path / "original/checkpoints/state/step_000002"
    state_dir.mkdir(parents=True)
    saved_cfg = compose_training_config([
        "task=robotwin_unified_flex_3cam_384_1e-4", "action_gradient=ramp",
        "action_gradient.ramp_end=0.6",
    ])
    OmegaConf.save(saved_cfg, tmp_path / "original/config.yaml")
    launch_cfg = compose_training_config([
        "task=robotwin_unified_flex_3cam_384_1e-4", f"action_gradient={group}",
        "resume_config_source=saved_with_overrides", f"resume={state_dir}",
        f"output_dir={tmp_path / 'resumed'}", "log_every=17", *leaf_overrides,
    ])
    restored = _maybe_swap_with_saved_config(launch_cfg)
    assert OmegaConf.is_dict(restored.action_gradient)
    assert restored.action_gradient.start_scale == 0.0
    assert restored.action_gradient.end_scale == 1.0
    assert restored.action_gradient.ramp_end == expected_end
    assert restored.log_every == 17
    assert restored.resume == str(state_dir)
    assert restored.output_dir == str(tmp_path / "resumed")
    ActionGradientSchedule(total_steps=1000, **dict(restored.action_gradient))


def test_real_trainer_accumulation_and_full_state_resume(
    cuda, tmp_path, monkeypatch, compose_training_config,
):
    """Compare resumed training with uninterrupted updates through the ramp."""
    from flexpi import runtime
    from flexpi.trainer import Wan22Trainer

    class TrainingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mot, self.inputs = make_case(cuda, hbridge=True)
            self.seen_scales = []

        def training_loss(self, sample):
            self.seen_scales.append(self.mot.action_gradient_scale)
            output = self.mot(**self.inputs)
            loss = sum(x.square().mean() for x in output.values())
            return loss, {}

        def save_checkpoint(self, path, optimizer=None, step=None):
            torch.save({"model": self.state_dict(), "step": step}, path)

    cfg = OmegaConf.create(dict(
        model={}, data={}, output_dir=str(tmp_path / "original"), learning_rate=1e-4,
        weight_decay=0.0, batch_size=2, num_workers=0, num_epochs=1,
        max_steps=6, log_every=0, save_every=2, eval_every=0,
        eval_num_inference_steps=1, gradient_accumulation_steps=2,
        max_grad_norm=1.0, seed=42, resume=None, mixed_precision="no",
        keep_last_n_states=5, keep_last_n_weights=5, wandb={"enabled": False},
        action_gradient=dict(start_scale=0.0, end_scale=1.0,
                             ramp_start=0.1, ramp_end=0.4),
    ))
    model = TrainingModel()
    trainer = Wan22Trainer(model, list(range(64)), cfg=cfg)
    OmegaConf.save(cfg, tmp_path / "original/config.yaml")
    trainer.train()
    expected = [trainer.action_gradient_schedule.value(step)
                for step in range(6) for _ in range(2)]
    assert model.seen_scales == expected
    state_dir = tmp_path / "original/checkpoints/state/step_000002"
    saved = json.loads((state_dir / "trainer_state.json").read_text())
    assert saved["global_step"] == 2
    assert saved["action_gradient_schedule"]["total_steps"] == 6
    final_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}

    launch_cfg = compose_training_config([
        "task=robotwin_unified_flex_3cam_384_1e-4", "action_gradient=ramp",
        "resume_config_source=saved_with_overrides", f"resume={state_dir}",
        f"output_dir={tmp_path / 'resumed'}", "log_every=1",
    ])
    resumed_model = TrainingModel()
    resumed_trainers = []

    def construct_trainer(**kwargs):
        resumed = Wan22Trainer(**kwargs)
        assert resumed.global_step == 2
        resumed_trainers.append(resumed)
        return resumed

    # Substitute only the expensive model/data construction; use the real
    # run_training -> saved-config merge -> trainer -> checkpoint-load path.
    monkeypatch.setattr(runtime, "instantiate", lambda *args, **kwargs: resumed_model)
    monkeypatch.setattr(runtime, "build_datasets", lambda data_cfg: (list(range(64)), None))
    monkeypatch.setattr(runtime, "Wan22Trainer", construct_trainer)
    runtime.run_training(launch_cfg)
    assert resumed_trainers[0].global_step == 6
    assert resumed_trainers[0].log_every == 1
    assert resumed_model.seen_scales == expected[4:]
    for name, parameter in resumed_model.named_parameters():
        torch.testing.assert_close(parameter, final_parameters[name], rtol=0, atol=0)


def test_training_entry_seeds_random_heads_before_construction(cuda, tmp_path, monkeypatch):
    from flexpi import runtime

    initial_weights = []

    def construct_head(config, model_dtype, device):
        head = torch.nn.Linear(8, 4, device=device, dtype=model_dtype)
        initial_weights.append(head.weight.detach().clone())
        return head

    class NoTraining:
        def __init__(self, **kwargs):
            pass

        def train(self):
            pass

    monkeypatch.setattr(runtime, "instantiate", construct_head)
    monkeypatch.setattr(runtime, "build_datasets", lambda cfg: ([], None))
    monkeypatch.setattr(runtime, "Wan22Trainer", NoTraining)
    monkeypatch.setattr(runtime, "_resolve_train_device", lambda: cuda)
    for index, seed in enumerate((42, 42, 43)):
        cfg = OmegaConf.create(dict(output_dir=str(tmp_path / str(index)),
                                    model={}, data={}, seed=seed, mixed_precision="no"))
        runtime.run_training(cfg)
    torch.testing.assert_close(initial_weights[0], initial_weights[1], rtol=0, atol=0)
    assert not torch.equal(initial_weights[0], initial_weights[2])

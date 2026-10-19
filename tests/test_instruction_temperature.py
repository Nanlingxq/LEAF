"""CPU-only checks on synthetic tensors; no EEG, real LEAF or checkpoint loaded."""
import hashlib
import csv
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import b_instruct_tuning as tuning
from load_config import build_train_config, load_yaml


class TinyLeaf(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tower = nn.Module()
        self.tower.tokenizer = nn.Linear(4, 4, bias=False)
        self.projection = nn.Linear(4, 4, bias=False)

    def forward(self, x, instruction):
        output = self.projection(self.tower.tokenizer(x) + instruction)
        return F.normalize(output, dim=-1), output


def make_module(temperature=None):
    train = dict(lr_base=1e-3, lr_scale_fast_params=1.0,
                 lr_scale_slow_params=0.2, max_epochs=100, warmup_epochs=2,
                 weight_decay=1e-3, cosine_annealing_lr_base=1.0,
                 cosine_annealing_lr_min=0.1, save_ckpt_every_n_epoch=10)
    if temperature is not None:
        train['temperature'] = temperature
    with patch.object(tuning, 'LEAF', TinyLeaf):
        return tuning.Instruct_Tuning_Module(
            SimpleNamespace(), SimpleNamespace(**train), torch.eye(4).numpy(),
            None, niter_per_epoch=4, seed=42)


class TemperatureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(3, 4)
        self.instruction = torch.randn(3, 4)
        self.y = torch.tensor([0, 1, 2])

    def test_legacy_config_uses_original_logits_and_two_groups(self):
        module = make_module()
        original = module.model(self.x, self.instruction)[0] @ module.prototypes
        torch.testing.assert_close(module.compute_logits(self.x, self.instruction), original,
                                   rtol=0, atol=0)
        self.assertIsNone(module.logit_scale)
        optimizers, _ = module.configure_optimizers()
        self.assertEqual(len(optimizers[0].param_groups), 2)

    def test_initial_scale_and_shared_train_validation_loss(self):
        module = make_module({'enabled': True})
        self.assertEqual(module.logit_scale.exp().item(), 1.0)
        with torch.no_grad():
            module.logit_scale.fill_(math.log(3.0))
        expected = F.cross_entropy(module.compute_logits(self.x, self.instruction), self.y)
        captured = {}
        module.log = lambda name, value, **kwargs: captured.update({name: value})
        train_loss = module.training_step((self.x, self.instruction, self.y), 0)
        module.validation_step((self.x, self.instruction, self.y), 0)
        torch.testing.assert_close(train_loss, expected)
        torch.testing.assert_close(captured['loss'], expected)
        raw = module.model(self.x, self.instruction)[0] @ module.prototypes
        torch.testing.assert_close(captured['loss_unscaled'], F.cross_entropy(raw, self.y))

    def test_scalar_gets_gradient_and_optimizer_update_without_decay(self):
        module = make_module({'enabled': True})
        optimizers, schedulers = module.configure_optimizers()
        optimizer = optimizers[0]
        self.assertEqual(len(optimizer.param_groups), 3)
        self.assertEqual(optimizer.param_groups[-1]['weight_decay'], 0.0)
        self.assertEqual(optimizer.param_groups[-1]['initial_lr'], 2e-4)
        ids = [id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn(id(module.logit_scale), ids)
        loss = module.training_step((self.x, self.instruction, self.y), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(module.logit_scale.grad))
        self.assertGreater(abs(module.logit_scale.grad.item()), 1e-8)
        # Override zero LR at the first warmup step for this synthetic update check.
        optimizer.param_groups[-1]['lr'] = 0.01
        before = module.logit_scale.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, module.logit_scale.detach()))
        self.assertEqual(schedulers[0]['interval'], 'step')

    def test_projection_bounds_and_fp32_under_autocast(self):
        module = make_module({'enabled': True})
        for value, expected in [(20.0, math.log(100.0)), (-20.0, math.log(0.01))]:
            with torch.no_grad():
                module.logit_scale.fill_(value)
            module.on_train_batch_end(None, None, 0)
            self.assertAlmostEqual(module.logit_scale.item(), expected, places=5)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logits = module.compute_logits(self.x, self.instruction)
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logits).all())

    def test_sidecar_and_legacy_model_loading(self):
        module = make_module({'enabled': True})
        module._trainer = SimpleNamespace(current_epoch=59, global_step=240)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'it060.ckpt'
            torch.save(module.model.state_dict(), checkpoint)
            module._save_temperature(checkpoint)
            sidecar = torch.load(str(checkpoint) + '.temperature.pt', weights_only=True)
            self.assertEqual(sidecar['epoch'], 60)
            self.assertEqual(sidecar['model_sha256'], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(sidecar['scale'].item(), 1.0)
            self.assertEqual(sidecar['temperature'].item(), 1.0)
            TinyLeaf(None).load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)

    def test_invalid_settings_rejected_and_yaml_enabled(self):
        for config in [{'enabled': 'false'}, {'enabled': True, 'scale_init': 0},
                       {'enabled': True, 'scale_min': 2, 'scale_max': 1},
                       {'enabled': True, 'lr': float('nan')}]:
            with self.assertRaises(ValueError):
                make_module(config)
        cfg = load_yaml(Path(__file__).resolve().parents[1] / 'configs/LEAF_mpnet.yaml')
        train = build_train_config(cfg, 'instruct')
        self.assertTrue(train.temperature['enabled'])
        self.assertEqual(train.temperature['scale_init'], 1.0)

    def test_epoch_hook_writes_history_and_paired_sidecar(self):
        module = make_module({'enabled': True})
        module._trainer = SimpleNamespace(current_epoch=9, global_step=40,
                                          global_rank=0, sanity_checking=False)
        module.configure_optimizers()
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'it010.ckpt'
            with patch.object(tuning, 'fn_save_ckpt', lambda epoch: checkpoint, create=True):
                module.on_train_epoch_end()
            with (Path(folder) / 'temperature_history.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]['epoch']), 10)
            self.assertEqual(float(rows[0]['scale']), 1.0)
            self.assertTrue(checkpoint.exists())
            self.assertTrue(Path(str(checkpoint) + '.temperature.pt').exists())


if __name__ == '__main__':
    unittest.main()

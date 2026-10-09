import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from cadrec import PointBertEncoder, UtoniaEncoder


class TinyPointBert(nn.Module):
    """Small external-backbone fixture; production checkpoint loading stays real."""

    def __init__(self, config):
        super().__init__()
        self.group_divider = nn.Linear(3, 3)
        self.transformer_q = nn.Linear(3, 4)
        self.transformer_q.trans_dim = 4
        self.num_group = config.dvae_config.num_group
        with torch.no_grad():
            self.group_divider.bias.fill_(float(Path(config.dvae_config.ckpt).read_text()))


def write_pointbert_assets(root, value, groups):
    config = root / 'cfgs/Mixup_models/Point-BERT.yaml'
    checkpoint = root / 'ckpt/Point-BERT.pth'
    dvae = root / 'ckpt/dVAE.pth'
    config.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    # Upstream YAML includes dataset files that the encoder must not try to load.
    config.write_text(
        'dataset:\n'
        '  train:\n'
        '    _base_: cfgs/dataset_configs/unavailable.yaml\n'
        'model:\n'
        '  dvae_config:\n'
        f'    num_group: {groups}\n'
        '    ckpt: unused.pth\n', encoding='utf-8')
    dvae.write_text(str(value + 1))
    state = {
        'group_divider.weight': torch.full((3, 3), value),
        'transformer_q.weight': torch.full((4, 3), value),
        'transformer_q.bias': torch.full((4,), value),
    }
    # The missing group bias is initialized from the separate dVAE fixture.
    torch.save(state, checkpoint)
    return config, checkpoint, dvae


def load_utonia_model(name, repo_id, ckpt_only=False, download_root=None):
    checkpoint = torch.load(name, weights_only=True)
    if ckpt_only:
        return checkpoint
    model = nn.Linear(3, 4)
    model.load_state_dict(checkpoint['state_dict'])
    return model


class EncoderPathTests(unittest.TestCase):
    def test_pointbert_root_overrides_and_reload_keep_selected_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = write_pointbert_assets(root / 'first package', 2.0, 5)
            second = write_pointbert_assets(root / 'other weights', 7.0, 9)
            builder = ModuleType('tools.builder')
            builder.model_builder = TinyPointBert
            tools = ModuleType('tools')
            tools.builder = builder
            modules = {'tools': tools, 'tools.builder': builder}
            env = {'HOME': str(root), 'POINTBERT_ROOT': '~/first package'}
            with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, modules), \
                    patch.object(sys, 'path', sys.path.copy()):
                default = PointBertEncoder(hidden_size=8, fusion_dim=8)
                self.assertEqual(default.num_groups, 5)
                torch.testing.assert_close(default.transformer.weight, torch.full((4, 3), 2.0))
                torch.testing.assert_close(default.group_divider.bias, torch.full((3,), 3.0))

                os.environ.update(POINTBERT_CONFIG=str(second[0]),
                                  POINTBERT_CHECKPOINT=str(second[1]),
                                  POINTBERT_DVAE_CHECKPOINT=str(second[2]))
                custom = PointBertEncoder(hidden_size=8, fusion_dim=8)
                self.assertEqual(custom.num_groups, 9)
                torch.testing.assert_close(custom.transformer.weight, torch.full((4, 3), 7.0))
                torch.testing.assert_close(custom.group_divider.bias, torch.full((3,), 8.0))

                explicit = PointBertEncoder(hidden_size=8, fusion_dim=8,
                                            pointbert_root=root / 'first package',
                                            config_path=first[0], ckpt_path=first[1],
                                            dvae_ckpt_path=first[2])
                torch.testing.assert_close(explicit.transformer.weight, torch.full((4, 3), 2.0))
                torch.testing.assert_close(explicit.group_divider.bias, torch.full((3,), 3.0))
                self.assertEqual(explicit.num_groups, 5)

                os.environ['POINTBERT_CHECKPOINT'] = str(first[1])
                custom.transformer.weight.zero_()
                custom.reload_backbone_weights()
                torch.testing.assert_close(custom.transformer.weight, torch.full((4, 3), 7.0))
                with self.assertRaisesRegex(FileNotFoundError, 'missing.pth'):
                    PointBertEncoder(hidden_size=8, ckpt_path=root / 'missing.pth')

    def test_utonia_local_checkpoint_precedence_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / 'first weights.pth', root / 'second weights.pth']
            for path, value in zip(paths, [3.0, 8.0]):
                torch.save({'config': {'dec_channels': [4], 'in_channels': 3},
                            'state_dict': {'weight': torch.full((4, 3), value),
                                           'bias': torch.full((4,), value)}}, path)
            utonia = ModuleType('utonia')
            utonia.model = SimpleNamespace(load=load_utonia_model)
            utonia.transform = SimpleNamespace(default=lambda **kwargs: None)
            env = {'HOME': str(root), 'UTONIA_CHECKPOINT': '~/first weights.pth',
                   'UTONIA_DOWNLOAD_ROOT': '~/cache'}
            with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, {'utonia': utonia}):
                default = UtoniaEncoder(hidden_size=8)
                torch.testing.assert_close(default.backbone.weight, torch.full((4, 3), 3.0))
                self.assertEqual(default.download_root, str(root / 'cache'))

                explicit = UtoniaEncoder(hidden_size=8, ckpt_path=paths[1],
                                         download_root=root / 'custom cache')
                torch.testing.assert_close(explicit.backbone.weight, torch.full((4, 3), 8.0))
                self.assertEqual(explicit.download_root, str(root / 'custom cache'))
                explicit.backbone.weight.zero_()
                explicit.reload_backbone_weights()
                torch.testing.assert_close(explicit.backbone.weight, torch.full((4, 3), 8.0))
                with self.assertRaisesRegex(FileNotFoundError, 'missing.pth'):
                    UtoniaEncoder(hidden_size=8, ckpt_path=root / 'missing.pth')


if __name__ == '__main__':
    unittest.main()

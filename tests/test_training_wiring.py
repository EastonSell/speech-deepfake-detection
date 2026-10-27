"""Exercise control flow with synthetic records and stubbed ML dependencies."""
import contextlib
import importlib
import io
import json
import numpy  # Keep NumPy loaded outside the temporary module patch.
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class TrainingWiringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.metadata = Path(self.tmp.name) / 'protocol'
        self.metadata.write_text('synthetic protocol')
        self.records = [{'trial_id': x, 'label': label} for x, label in [('a','fake'), ('b','real'), ('c','fake')]]
        self.dataset = types.SimpleNamespace(records=self.records, protocol_path=self.metadata)
        # Dataset must support len() just like the real dataset.
        class Dataset:
            def __init__(inner, records, path): inner.records, inner.protocol_path = records, path
            def __len__(inner): return len(inner.records)
        self.dataset = Dataset(self.records, self.metadata)
        self.loads = []
        torch = types.ModuleType('torch')
        torch.no_grad = lambda: lambda f: f
        torch.manual_seed = lambda seed: None
        torch.device = lambda value: value
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.Generator = lambda: types.SimpleNamespace(manual_seed=lambda seed: seed)
        torch.optim = types.SimpleNamespace(Adam=lambda *a, **kw: object())
        torch.save = lambda state, path: Path(path).write_bytes(b'stub weights')
        nn = types.ModuleType('torch.nn'); nn.CrossEntropyLoss = lambda **kw: object()
        data = types.ModuleType('torch.utils.data')
        def loader(ds, **kw): self.loads.append(ds); return []
        data.DataLoader = loader
        data.random_split = lambda ds, lengths, generator: (types.SimpleNamespace(indices=[2,0]), types.SimpleNamespace(indices=[1]))
        utils = types.ModuleType('torch.utils'); utils.data = data
        dataset_mod = types.ModuleType('pm.dataset.dataloader')
        dataset_mod.ASVspoofAudioDataset = object; dataset_mod.LABEL_TO_ID = {'fake': 0, 'real': 1}
        model_mod = types.ModuleType('pm.model.model')
        class Model:
            def __init__(self, **kw): pass
            def to(self, device): return self
            def parameters(self): return []
            def state_dict(self): return {}
            def train(self): pass
        model_mod.AudioClassifier = Model
        modules = {'torch':torch,'torch.nn':nn,'torch.utils':utils,'torch.utils.data':data,
                   'pm.dataset.dataloader':dataset_mod,'pm.model.model':model_mod}
        self.modpatch = patch.dict(sys.modules, modules); self.modpatch.start(); self.addCleanup(self.modpatch.stop)
        sys.modules.pop('pm.train_model', None)
        self.addCleanup(lambda: sys.modules.pop('pm.train_model', None))
        self.train = importlib.import_module('pm.train_model')
        self.train.build_dataset = lambda args: self.dataset
        self.train.class_weights = lambda records, device: self.weight_records.append(records)
        self.weight_records = []

    def args(self, frac, epochs=0):
        return types.SimpleNamespace(seed=5, device='cpu', example=False, metadata=str(self.metadata),
            limit=None, val_frac=frac, batch_size=2, num_workers=0, balance=True, lr=.01,
            epochs=epochs, out=str(Path(self.tmp.name)/'model.pt'))

    def test_split_indices_recorded_and_weights_only_from_training(self):
        with contextlib.redirect_stdout(io.StringIO()): self.train.train(self.args(.34))
        sidecar = next(Path(self.tmp.name).glob('model.pt.split.*.json'))
        manifest = json.loads(sidecar.read_text())
        self.assertEqual([x['index'] for x in manifest['train']], [2,0])
        self.assertEqual([x['index'] for x in manifest['validation']], [1])
        self.assertEqual(self.weight_records[0], [self.records[2], self.records[0]])

    def test_zero_validation_never_calls_evaluate(self):
        with patch.object(self.train, 'evaluate', side_effect=AssertionError('evaluation called')):
            with contextlib.redirect_stdout(io.StringIO()) as output: self.train.train(self.args(0, epochs=1))
        self.assertIn('validation disabled', output.getvalue())
        self.assertEqual(len(self.loads), 1)
        m=json.loads(next(Path(self.tmp.name).glob('model.pt.split.*.json')).read_text())
        self.assertEqual(m['validation'], [])

    def test_single_class_eer_is_unavailable(self):
        eer, threshold = self.train.compute_eer([.2,.3], [1,1])
        self.assertTrue(eer != eer and threshold != threshold)

    def test_nonpositive_limit_is_rejected(self):
        args=self.args(.2); args.limit=-1
        with self.assertRaisesRegex(ValueError, '--limit'): self.train.train(args)

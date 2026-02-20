#!/usr/bin/env python3
"""Unit and smoke tests for L1NanoDataset."""

import os
import unittest

try:
    import awkward as ak
    import torch
except ImportError:
    ak = None
    torch = None

try:
    from tools.training.InputDataset import L1NanoDataset
except Exception:
    L1NanoDataset = None


class _Obj:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestL1NanoDataset(unittest.TestCase):
    def setUp(self):
        if ak is None or torch is None or L1NanoDataset is None:
            self.skipTest("Missing required dependencies for L1NanoDataset tests")
        self.dataset = L1NanoDataset(dataset=[])

    def test_matching_labels_delta_r_and_indices(self):
        event = _Obj(
            stub=_Obj(
                offeta1=ak.Array([0.10, 1.80]),
                offphi1=ak.Array([0.20, 2.80]),
                tfLayer=ak.Array([1, 2]),
            ),
            GenPart=_Obj(
                pdgId=ak.Array([13]),
                statusFlags=ak.Array([1 << 13]),
                pt=ak.Array([10.0]),
                etaSt2=ak.Array([0.12]),
                phiSt2=ak.Array([0.25]),
                eta=ak.Array([0.12]),
                phi=ak.Array([0.25]),
                dXY=ak.Array([1.0]),
                lXY=ak.Array([10.0]),
            ),
        )

        labels, delta_r, matched_idx = self.dataset._match_stubs_to_genpart(event)

        self.assertEqual(labels.tolist(), [1, 0])
        self.assertEqual(matched_idx.tolist(), [0, -1])
        self.assertLess(delta_r[0].item(), self.dataset.dR_threshold)
        self.assertEqual(delta_r[1].item(), 999.0)

    def test_matching_no_muons_returns_minus_one(self):
        event = _Obj(
            stub=_Obj(
                offeta1=ak.Array([0.10, -0.20]),
                offphi1=ak.Array([0.20, -1.50]),
                tfLayer=ak.Array([1, 2]),
            ),
            GenPart=_Obj(
                pdgId=ak.Array([]),
                statusFlags=ak.Array([]),
                pt=ak.Array([]),
                etaSt2=ak.Array([]),
                phiSt2=ak.Array([]),
                eta=ak.Array([]),
                phi=ak.Array([]),
                dXY=ak.Array([]),
                lXY=ak.Array([]),
            ),
        )

        labels, delta_r, matched_idx = self.dataset._match_stubs_to_genpart(event)

        self.assertEqual(labels.tolist(), [-1, -1])
        self.assertEqual(matched_idx.tolist(), [-1, -1])
        self.assertEqual(delta_r.tolist(), [999.0, 999.0])

    def test_edge_construction_with_fallback_and_labels(self):
        stub_features = torch.tensor(
            [
                [1.0, 0.00, 0.00],
                [2.0, 2.00, 2.00],
                [3.0, 0.20, 0.20],
            ],
            dtype=torch.float32,
        )
        matched_idx = torch.tensor([4, 1, 4], dtype=torch.long)

        edge_index, edge_attr, edge_y = self.dataset._create_edges_by_layer(stub_features, matched_idx)

        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(edge_attr.shape[1], 2)
        self.assertGreaterEqual(edge_index.shape[1], 1)
        self.assertIn(1.0, edge_y.tolist())

    def test_optional_root_smoke(self):
        root_file = os.environ.get("L1NANO_TEST_ROOT", "")
        if not root_file or not os.path.exists(root_file):
            self.skipTest("Set L1NANO_TEST_ROOT to run ROOT NanoAOD smoke test")

        dataset = L1NanoDataset(
            root_dir=root_file,
            tree_name="Events",
            max_events=10,
            debug=False,
        )
        self.assertGreaterEqual(len(dataset), 0)


if __name__ == "__main__":
    unittest.main()

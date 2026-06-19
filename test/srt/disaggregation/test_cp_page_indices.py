import importlib.util
import sys
import types
import unittest
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _install_utils_import_stubs():
    for name in ["sglang", "sglang.srt", "sglang.srt.disaggregation"]:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules.setdefault(name, module)

    base_module = types.ModuleType("sglang.srt.disaggregation.base")

    class KVPoll(IntEnum):
        Bootstrapping = 0
        WaitingForInput = 1
        Transferring = 2
        Success = 3
        Failed = 4

    base_module.KVPoll = KVPoll
    sys.modules["sglang.srt.disaggregation.base"] = base_module

    environ_module = types.ModuleType("sglang.srt.environ")

    class _EnvField:
        def get(self):
            return 0

    environ_module.envs = SimpleNamespace(SGLANG_TEST_DISAGG_FAILURE_PROB=_EnvField())
    sys.modules["sglang.srt.environ"] = environ_module

    utils_module = types.ModuleType("sglang.srt.utils")
    utils_module.is_npu = lambda: False
    sys.modules["sglang.srt.utils"] = utils_module


def _load_disaggregation_utils():
    _install_utils_import_stubs()
    repo_root = Path(__file__).resolve().parents[3]
    utils_path = repo_root / "python" / "sglang" / "srt" / "disaggregation" / "utils.py"
    spec = importlib.util.spec_from_file_location(
        "sglang_srt_disaggregation_utils_under_test", utils_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_utils = _load_disaggregation_utils()
filter_kv_indices_for_cp_rank = _utils.filter_kv_indices_for_cp_rank
page_indices_to_cp_rank_page_indices = _utils.page_indices_to_cp_rank_page_indices


class TestCpPageIndices(unittest.TestCase):
    def test_filter_by_request_position_not_page_value(self):
        page_indices = np.asarray([100, 7, 42, 9, 88], dtype=np.int32)

        actual = page_indices_to_cp_rank_page_indices(
            page_indices, total_pages=5, cp_rank=1, cp_size=2
        )

        np.testing.assert_array_equal(actual, np.asarray([9, 88], dtype=np.int32))

    def test_filter_chunk_uses_global_request_page_offset(self):
        kv_mgr = SimpleNamespace(attn_cp_rank=1, attn_cp_size=2)
        chunk = np.asarray([42, 9, 88], dtype=np.int32)

        actual_indices, actual_slice = filter_kv_indices_for_cp_rank(
            kv_mgr, chunk, slice(2, 5), total_pages=5
        )

        np.testing.assert_array_equal(
            actual_indices, np.asarray([9, 88], dtype=np.int32)
        )
        self.assertEqual(actual_slice, slice(3, 5))

    def test_chunk_outside_rank_returns_empty_slice_at_chunk_start(self):
        kv_mgr = SimpleNamespace(attn_cp_rank=0, attn_cp_size=2)
        chunk = np.asarray([9, 88], dtype=np.int32)

        actual_indices, actual_slice = filter_kv_indices_for_cp_rank(
            kv_mgr, chunk, slice(3, 5), total_pages=5
        )

        self.assertEqual(actual_indices.size, 0)
        self.assertEqual(actual_slice, slice(3, 3))


if __name__ == "__main__":
    unittest.main()

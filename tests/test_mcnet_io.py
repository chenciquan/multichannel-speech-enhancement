"""Verify physical gain/length of the compatibility IO, independent of weights.

These are numerical fixtures, never evidence about human speech preservation.
They require the inference environment and byte-verified upstream helper file.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_UPSTREAM = (ROOT / "third_party/McNet/src/util/acoustic_utils.py").is_file()


@unittest.skipUnless(HAS_TORCH and HAS_UPSTREAM, "Requires inference dependencies and upstream checkout")
class McNetIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import runpy
        import torch
        cls.torch = torch
        adapter = runpy.run_path(str(ROOT / "scripts/mcnet_infer.py"), run_name="io_test")
        helper_path = ROOT / "third_party/McNet/src/util/acoustic_utils.py"
        expected = adapter["SOURCE_BLOBS"]["src/util/acoustic_utils.py"]
        if adapter["file_hashes"](helper_path)["git_blob_sha1"] != expected:
            raise RuntimeError("Upstream acoustic helper differs from pinned source")
        cls.adapter = staticmethod(adapter["reconstruct_waveform"])
        cls.helper = runpy.run_path(str(helper_path), run_name="upstream_acoustics_test")

    def check_gain(self, gain):
        torch = self.torch
        generator = torch.Generator().manual_seed(137)
        x = torch.randn(1, 6, 4099, generator=generator) * 0.07
        target = x[:, 4] * gain
        io = SimpleNamespace(ft_len=512, ft_hop=256, ref_chn_idx=4,
                             window=torch.hann_window(512))
        xr = torch.stft(x[:, 4], 512, 256, window=io.window, return_complex=True)
        yr = torch.stft(target, 512, 256, window=io.window, return_complex=True)
        mask = self.helper["get_complex_ideal_ratio_mask"](
            torch.view_as_real(xr), torch.view_as_real(yr))
        output = self.adapter(SimpleNamespace(io=io), self.helper["icIRM"], x,
                              torch.view_as_complex(mask.contiguous()), x.shape[-1])
        self.assertEqual(tuple(output.shape), (1, 4099))
        torch.testing.assert_close(output, target, atol=2e-6, rtol=2e-4)

    def test_identity_keeps_reference_scale_and_non_hop_aligned_length(self):
        self.check_gain(1.0)

    def test_known_quarter_gain_does_not_get_an_extra_normalization(self):
        self.check_gain(0.25)


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""70_time_pair_index.py 的回归测试(纯标准库,无需 torch/数据)。

跑法:`python3 -m unittest tests.test_time_pair_index -v` 或直接 `python3 tests/test_time_pair_index.py`。

为什么值得有:阶段 5 的样本量(train 2640 / valid 360 / test 715)是写进预登记文档的,
换划分、换 alpha 数、改边界归属规则都会改这个分母;而这类改动**不会报错**,只会
让分母悄悄变掉——正是 10-31 那次 spread 分母事故的同一类风险。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import datetime as dt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "outline", "70_time_pair_index.py")


def make_frames(dirpath, n_hours, start=dt.datetime(2020, 7, 1), minute_step=10):
    """造 n_hours 小时的帧;每小时含 6 帧(00/10/20/30/40/50 分)。"""
    os.makedirs(dirpath, exist_ok=True)
    for i in range(n_hours):
        t = start + dt.timedelta(hours=i)
        for m in range(0, 60, minute_step):
            s = (t + dt.timedelta(minutes=m)).strftime("%Y%m%dT%H%M%S")
            open(os.path.join(dirpath, "c_myj_%s.npz" % s), "w").close()


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def run_index(fine_dir, split_path, out_path, extra=()):
    cmd = [sys.executable, SCRIPT, "--fine_dir", fine_dir, "--split_json", split_path,
           "--out_json", out_path] + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError("脚本失败:\n" + r.stdout + r.stderr)
    with open(out_path) as f:
        return json.load(f), r.stdout


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="t5_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def path(self, *a):
        return os.path.join(self.tmp, *a)


class TestShape(TmpCase):
    def test_five_alphas_and_pair_wise_samples(self):
        """一对 -> 5 个样本,alpha = 1/6..5/6,且不被非整点帧污染。"""
        make_frames(self.path("fine"), 10)
        stamps = [(dt.datetime(2020, 7, 1) + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S")
                  for i in range(10)]
        write_json(self.path("split.json"), {"hours": {"train": stamps}})
        d, _ = run_index(self.path("fine"), self.path("split.json"), self.path("p.json"))
        self.assertEqual(d["stats"]["n_hour_frames"], 10)          # 只数整点,不数 10/20/...分
        self.assertEqual(d["stats"]["n_pairs"]["train"], 9)        # 末小时无 T+1
        self.assertEqual(d["stats"]["n_samples"]["train"], 45)     # 9 对 × 5
        self.assertEqual(len(d["alphas"]), 5)
        for a, want in zip(d["alphas"], [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]):
            self.assertAlmostEqual(a, want, places=6)
        first = d["splits"]["train"][0]
        self.assertTrue(first["t0"].endswith("20200701T000000.npz"))
        self.assertTrue(first["t1"].endswith("20200701T010000.npz"))
        self.assertAlmostEqual(first["alpha"], 1 / 6, places=6)
        self.assertEqual(d["stats"]["missing_next_hour"], 1)

    def test_n_alpha_option(self):
        make_frames(self.path("fine"), 4)
        stamps = [(dt.datetime(2020, 7, 1) + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S")
                  for i in range(4)]
        write_json(self.path("split.json"), {"hours": {"train": stamps}})
        d, _ = run_index(self.path("fine"), self.path("split.json"), self.path("p.json"),
                         extra=["--n_alpha", "2"])
        self.assertEqual(d["stats"]["n_samples"]["train"], 3 * 2)


class TestSplitFormats(TmpCase):
    def test_hours_dict_lists(self):
        """真实格式:hours 是 子集 -> 完整时间戳列表。"""
        make_frames(self.path("fine"), 6)
        base = dt.datetime(2020, 7, 1)
        hs = [(base + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S") for i in range(6)]
        write_json(self.path("split.json"), {"hours": {"train": hs[:4], "test": hs[4:]}})
        d, _ = run_index(self.path("fine"), self.path("split.json"), self.path("p.json"))
        self.assertEqual(d["stats"]["n_pairs"]["train"], 4)
        self.assertEqual(d["stats"]["n_pairs"]["test"], 1)   # 05 时在 test,06 时不存在 -> 丢掉

    def test_blocks_day_granularity(self):
        """兜底格式:blocks 按天给 start/end,要展开到当天 23 时。"""
        make_frames(self.path("fine"), 30)
        write_json(self.path("split.json"),
                   {"blocks": [{"start": "2020-07-01", "end": "2020-07-01", "split": "train"},
                               {"start": "2020-07-02", "end": "2020-07-02", "split": "test"}]})
        d, _ = run_index(self.path("fine"), self.path("split.json"), self.path("p.json"))
        self.assertEqual(d["stats"]["n_pairs"]["train"], 24)   # 00..23 时,23 时那对跨到次日
        self.assertEqual(d["stats"]["n_pairs"]["test"], 5)     # 02 日 00..05 时里 05 时无 T+1


class TestBoundary(TmpCase):
    def test_cross_split_pair_counted_by_start(self):
        """跨子集的边界对按起点归属,并被计数报出(不许静默)。"""
        make_frames(self.path("fine"), 8)
        base = dt.datetime(2020, 7, 1)
        hs = [(base + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S") for i in range(8)]
        write_json(self.path("split.json"), {"hours": {"train": hs[:4], "valid": hs[4:]}})
        d, _ = run_index(self.path("fine"), self.path("split.json"), self.path("p.json"))
        # 03 时(train)的下一小时 04 在 valid -> 1 个跨界对,仍归 train
        self.assertEqual(d["stats"]["cross_split_pairs"], 1)
        self.assertEqual(d["stats"]["n_pairs"]["train"], 4)
        self.assertEqual(d["stats"]["n_pairs"]["valid"], 3)

    def test_hours_not_in_split_reported(self):
        make_frames(self.path("fine"), 6)
        base = dt.datetime(2020, 7, 1)
        hs = [(base + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S") for i in range(2)]
        write_json(self.path("split.json"), {"hours": {"train": hs}})
        d, _ = run_index(self.path("fine"), self.path("split.json"), self.path("p.json"))
        # 06 帧里 2 帧在划分、1 帧(最后一小时)先被 missing_next_hour 吃掉,
        # 剩下 3 帧才算"不在划分里"——顺序是脚本里定死的,这里锁住它
        self.assertEqual(d["stats"]["missing_next_hour"], 1)
        self.assertEqual(d["stats"]["hours_not_in_split"], 3)


class TestJulyScale(TmpCase):
    def test_expected_counts_2640_360_715(self):
        """预登记文档里写死的分母:train 2640 / valid 360 / test 715。"""
        make_frames(self.path("fine"), 744)
        base = dt.datetime(2020, 7, 1)
        hs = [(base + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S") for i in range(744)]
        write_json(self.path("split.json"),
                   {"hours": {"train": hs[:528], "valid": hs[528:600], "test": hs[600:]}})
        d, _ = run_index(self.path("fine"), self.path("split.json"), self.path("p.json"))
        self.assertEqual(d["stats"]["n_samples"], {"train": 2640, "valid": 360, "test": 715})
        self.assertEqual(d["stats"]["n_pairs"], {"train": 528, "valid": 72, "test": 143})
        self.assertEqual(d["stats"]["missing_next_hour"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

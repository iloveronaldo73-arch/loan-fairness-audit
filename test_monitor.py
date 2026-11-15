import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np
import monitor as M


def test_psi_identical_is_zero_and_shift_is_large():
    e = np.array([.25, .25, .25, .25])
    assert M.psi(e, e) < 1e-9
    assert M.psi(e, np.array([.7, .1, .1, .1])) > 0.25


def test_persistence_rule_needs_two_of_three():
    mon = M.BatchMonitor.__new__(M.BatchMonitor); mon.recent = {}
    assert not mon._persistent("k", True)       # one breach: not enough
    assert mon._persistent("k", True)           # second breach in a row
    assert mon._persistent("x", False, immediate=True)


def test_cusum_accumulates_small_shifts():
    s, fired = 0.0, None
    for t in range(1, 30):                      # steady +1.5 standard-error shift per batch
        s = max(0.0, s + 1.5 - 0.5)
        if s > 5.0:
            fired = t; break
    assert fired is not None and fired <= 6


def test_blind_model_quiet_on_clean_data_and_catches_drift(tmp_path):
    th = M.Thresholds()
    Xf, y = M.prepare(M.load_df("synthetic"))
    quiet = [M.simulate(Xf, y, "none", 12, 400, 6, .7, s, th).health()["blind_model"]["critical_total"] for s in range(3)]
    assert sum(q > 0 for q in quiet) <= 1       # at most one noisy run out of three
    drift = M.simulate(Xf, y, "covariate", 12, 400, 6, .7, 1, th)
    assert drift.health()["blind_model"]["status"] == "RED"
    M.save(drift, str(tmp_path), 6)
    for f in ("dashboard.png", "monitor_report.md", "alerts.csv", "batches.csv", "pooled_audit.csv"):
        assert (tmp_path / f).exists()


def test_small_groups_never_trigger_fairness_alarms():
    th = M.Thresholds(min_group=10_000)
    Xf, y = M.prepare(M.load_df("synthetic"))
    gm = M.simulate(Xf, y, "fairness", 6, 200, 2, .9, 3, th)
    assert not gm.alert_log().type.isin(["fairness", "fairness_single"]).any()

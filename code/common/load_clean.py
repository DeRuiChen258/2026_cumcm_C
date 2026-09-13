
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "clean"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_float(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def load_config(path: Path | None = None) -> dict:
    """Minimal YAML reader (2-level mappings + inline lists) — keeps dependencies fixed."""
    path = path or (ROOT / "code" / "common" / "config.yaml")
    out: dict = {}
    stack: list[tuple[int, dict]] = [(0, out)]
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, _, value = raw.strip().partition(":")
        while stack and indent < stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = value.strip()
        if value == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent + 2, child))
        elif value.startswith("[") and value.endswith("]"):
            body = value[1:-1].strip()
            parent[key] = [] if not body else [
                float(v.strip()) if _is_float(v.strip()) else v.strip() for v in body.split(",")
            ]
        elif _is_float(value):
            parent[key] = float(value)
        else:
            parent[key] = value
    return out


@dataclass(frozen=True)
class Panel:
    """Immutable clean panel handed to the modelling layer."""

    dates: tuple[str, ...]
    slots: tuple[str, ...]
    pv: np.ndarray          # 365x144, NaN only where quarantined (core night)
    pv_raw: np.ndarray
    load: np.ndarray
    price: np.ndarray
    pv_forecast: np.ndarray  # 365x4x24
    day1: np.ndarray         # 144x3 -> price, load, pv forecast (附件1)
    windows: dict[str, np.ndarray]
    sigma: dict[str, np.ndarray]
    quarantine_count: int
    audit_records: int
    meta: dict
    config: dict

    def pv_filled(self) -> np.ndarray:
        """PV with quarantined core-night cells read as 0 (physically certain at night)."""
        return np.nan_to_num(self.pv, nan=0.0)

    def net_load(self) -> np.ndarray:
        """Actual net load N = L - PV (365x144)."""
        return self.load - self.pv_filled()

    def day_index(self, iso: str) -> int:
        return self.dates.index(iso)


def load_panel(clean_dir: Path | None = None, verify_hashes: bool = True) -> Panel:
    clean_dir = Path(clean_dir or CLEAN_DIR)
    manifest = json.loads((clean_dir / "manifest.json").read_text(encoding="utf-8"))
    if verify_hashes:
        for name, want in manifest["outputs_sha256"].items():
            if _sha256(clean_dir / name) != want:
                raise RuntimeError(f"clean bundle checksum mismatch: {name}")

    pv_raw = np.load(clean_dir / "pv_actual_raw.npy")
    pv = np.load(clean_dir / "pv_actual.npy")
    load = np.load(clean_dir / "load.npy")
    price = np.load(clean_dir / "price.npy")
    pv_forecast = np.load(clean_dir / "pv_forecast.npy")
    day1 = np.load(clean_dir / "day1.npy")
    dates = tuple(json.loads((clean_dir / "dates.json").read_text(encoding="utf-8"))["dates"])
    slots = tuple(json.loads((clean_dir / "slots.json").read_text(encoding="utf-8"))["slots"])

    for name, arr, shape in (
        ("pv", pv, (365, 144)),
        ("load", load, (365, 144)),
        ("price", price, (365, 144)),
        ("pv_forecast", pv_forecast, (365, 4, 24)),
        ("day1", day1, (144, 3)),
    ):
        if arr.shape != shape or arr.dtype != np.float64:
            raise RuntimeError(f"clean bundle contract violation on {name}: {arr.shape} {arr.dtype}")

    windows = _read_csv_columns(clean_dir / "day_windows.csv")
    sigma = _read_csv_columns(clean_dir / "sigma.csv")
    with (clean_dir / "quarantine.csv").open(encoding="utf-8") as fh:
        quarantine_rows = sum(1 for _ in fh) - 1

    check_invariants(pv, load, price, windows)
    meta = json.loads((clean_dir / "meta.json").read_text(encoding="utf-8"))
    return Panel(
        dates=dates,
        slots=slots,
        pv=pv,
        pv_raw=pv_raw,
        load=load,
        price=price,
        pv_forecast=pv_forecast,
        day1=day1,
        windows=windows,
        sigma=sigma,
        quarantine_count=int(quarantine_rows),
        audit_records=int(manifest["audit_records"]),
        meta=meta,
        config=load_config(),
    )


def _read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, list] = {k: [] for k in rows[0].keys()}
    for row in rows:
        for key, value in row.items():
            if key in ("date", "sunrise", "sunset", "slot_label"):
                out[key].append(value)
                continue
            try:
                out[key].append(float(value))
            except (TypeError, ValueError):
                out[key].append(value)
    return {k: np.array(v) for k, v in out.items()}


def check_invariants(pv, load, price, windows) -> None:
    """Re-assert I1..I5 (read-only; any violation aborts the pipeline)."""
    # day_windows.csv stores 1-based slot indices (slot k = right endpoint 10k min);
    # the matrices are 0-based, hence the -1.
    k_rise = windows["k_rise"].astype(int) - 1
    k_set = windows["k_set"].astype(int) - 1
    for d in range(pv.shape[0]):
        core = np.ones(pv.shape[1], dtype=bool)
        core[max(k_rise[d] - 1, 0): k_set[d] + 2] = False
        slice_ = pv[d][core]
        finite = slice_[~np.isnan(slice_)]
        if finite.size and np.any(finite != 0.0):
            raise RuntimeError(f"I1 violated on day {d}")
    mid = pv[:, 47:96]
    finite_mid = mid[~np.isnan(mid)]
    if finite_mid.size and (np.any(finite_mid == 0.0) or np.any(finite_mid < 0.0)):
        raise RuntimeError("I2 violated (08:00-16:00 contains zeros/negatives)")
    if not np.all(price > 0.0) or not np.all(load > 0.0):
        raise RuntimeError("I5 violated (price/load must stay positive)")
    if load.min() < 1500.0 or load.max() > 8000.0:
        raise RuntimeError("I5 violated (load outside [1500, 8000] kW)")

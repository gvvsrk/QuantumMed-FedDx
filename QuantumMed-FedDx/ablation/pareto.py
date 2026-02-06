from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json

@dataclass
class ParetoConfig:
    perf_key: str = "test_macro_f1"
    complexity_key: str = "approx_gate_count"
    min_perf: Optional[float] = None
    max_complexity: Optional[float] = None

def _get_nested(d: Dict[str, Any], path: str):
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur

def load_runs(jsonl_path: str) -> List[Dict[str, Any]]:
    runs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                runs.append(json.loads(line))
    return runs

def pareto_front(runs: List[Dict[str, Any]], cfg: ParetoConfig) -> List[Dict[str, Any]]:
    candidates = []
    for r in runs:
        perf = _get_nested(r, f"evaluation.{cfg.perf_key}")
        comp = _get_nested(r, f"evaluation.quantum_complexity.{cfg.complexity_key}")
        if perf is None or comp is None:
            continue
        perf = float(perf); comp = float(comp)
        if cfg.min_perf is not None and perf < cfg.min_perf:
            continue
        if cfg.max_complexity is not None and comp > cfg.max_complexity:
            continue
        rr = dict(r); rr["_perf"] = perf; rr["_comp"] = comp
        candidates.append(rr)

    candidates.sort(key=lambda x: (x["_comp"], -x["_perf"]))
    frontier = []
    best_perf = -1e18
    for r in candidates:
        if r["_perf"] > best_perf:
            frontier.append(r); best_perf = r["_perf"]
    return frontier

def summarize_pareto(frontier: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in frontier:
        out.append({
            "ablation_point": r.get("ablation_point", {}),
            "perf": r.get("_perf"),
            "complexity": r.get("_comp"),
            "quantum_complexity": _get_nested(r, "evaluation.quantum_complexity") or {},
        })
    return out

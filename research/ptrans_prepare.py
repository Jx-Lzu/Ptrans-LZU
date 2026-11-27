#!/usr/bin/env python
"""Prepare PTrans experimental NMR data and deterministic splits.

Requirements:
    pip install pandas numpy rdkit

Example:
    python ptrans_prepare.py --input data/raw/experimental.csv.gz --outdir prepared --seed 42
"""
import argparse
import gzip
import ast
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parent

NUCLEI = ("1H_NMR", "13C_NMR", "31P_NMR")

def literal(v):
    if not isinstance(v, str):
        return None
    try:
        return ast.literal_eval(v)
    except Exception:
        return None

def numeric_values(v):
    if v is None:
        return []
    if isinstance(v, (int, float, np.integer, np.floating)):
        return [float(v)]
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            out.extend(numeric_values(x))
        return out
    if isinstance(v, str):
        try:
            parsed = ast.literal_eval(v)
            if isinstance(parsed, (list, tuple, int, float)):
                return numeric_values(parsed)
        except Exception:
            pass
        return [float(x) for x in re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", v
        )]
    return []

def coupling_values(v):
    return numeric_values(v)

def integral_value(v):
    if v is None:
        return None
    found = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(v))
    return float(found[0]) if found else None

def to_jsonable(x):
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x

def normalize_peak(col, item):
    if col == "1H_NMR":
        mult = item[0] if len(item) > 0 else None
        coupling_raw = item[1] if len(item) > 1 else None
        integral = item[2] if len(item) > 2 else None
        start = float(item[3]) if len(item) > 3 and isinstance(item[3], (int, float)) else None
        end = float(item[4]) if len(item) > 4 and isinstance(item[4], (int, float)) else None
        shifts = [x for x in (start, end) if x is not None]
        return {
            "nucleus": "1H",
            "multiplicity": str(mult) if mult is not None else None,
            "integral_label": str(integral) if integral is not None else None,
            "integral_protons": integral_value(integral),
            "shift_start_ppm": start,
            "shift_end_ppm": end,
            "shift_min_ppm": min(shifts) if shifts else None,
            "shift_max_ppm": max(shifts) if shifts else None,
            "coupling_hz": coupling_values(coupling_raw),
            "coupling_raw": to_jsonable(coupling_raw),
            "raw_peak": to_jsonable(item),
        }
    shift_raw = item[0] if len(item) > 0 else None
    shifts = numeric_values(shift_raw)
    mult = item[1] if len(item) > 1 else None
    coupling_raw = item[2] if len(item) > 2 else None
    return {
        "nucleus": "13C" if col == "13C_NMR" else "31P",
        "multiplicity": str(mult) if mult is not None else None,
        "shift_min_ppm": min(shifts) if shifts else None,
        "shift_max_ppm": max(shifts) if shifts else None,
        "shift_raw_values_ppm": shifts,
        "coupling_hz": coupling_values(coupling_raw),
        "coupling_raw": to_jsonable(coupling_raw),
        "raw_peak": to_jsonable(item),
    }

def find_outlier(col, peak):
    if col == "1H_NMR":
        vals = [x for x in (peak.get("shift_start_ppm"), peak.get("shift_end_ppm")) if x is not None]
        low, high = -1, 20
    else:
        vals = peak.get("shift_raw_values_ppm", [])
        low, high = (-20, 300) if col == "13C_NMR" else (-300, 300)
    if not vals:
        return None
    mn, mx = min(vals), max(vals)
    issues = []
    if mn < low:
        issues.append(f"shift_below_{low}_ppm")
    if mx > high:
        issues.append(f"shift_above_{high}_ppm")
    if issues:
        return {"issue_code": ";".join(issues), "min_shift": mn, "max_shift": mx}
    return None

def grouped_greedy_split(data, group_col, seed=42, ratios=(0.8, 0.1, 0.1)):
    rng = np.random.default_rng(seed)
    categories = sorted(data["p_category"].unique())
    agg = {"n": ("sample_id", "size")}
    for category in categories:
        agg[f"cat_{category}"] = ("p_category", lambda s, c=category: int((s == c).sum()))
    groups = data.groupby(group_col, sort=False).agg(**agg).reset_index()
    groups["random_tiebreak"] = rng.random(len(groups))
    groups = groups.sort_values(["n", "random_tiebreak"], ascending=[False, True]).reset_index(drop=True)
    total_by_category = data["p_category"].value_counts().to_dict()
    target = np.array([[total_by_category.get(c, 0) * r for c in categories] for r in ratios], dtype=float)
    current = np.zeros_like(target)
    names = ("train", "val", "test")
    assignment = {}
    for _, group in groups.iterrows():
        vector = np.array([group[f"cat_{c}"] for c in categories], dtype=float)
        costs = []
        for split_index in range(3):
            delta = np.sum(
                (2 * vector * (current[split_index] - target[split_index]) + vector**2)
                / np.maximum(target[split_index], 1)
            )
            costs.append(delta)
        chosen = min(range(3), key=lambda i: (costs[i], i))
        assignment[group[group_col]] = names[chosen]
        current[chosen] += vector
    return data[group_col].map(assignment)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/experimental.csv.gz", help="CSV or CSV.gz; relative paths use package root")
    parser.add_argument("--outdir", default="prepared")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = ROOT / args.input
    outdir = ROOT / args.outdir
    if outdir.resolve() == (ROOT / "data").resolve():
        raise ValueError("Use a separate output directory; bundled data are frozen.")
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    required = {"SMILES", "molecular_weight", "1H_NMR", "13C_NMR", "31P_NMR"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    canonical, formula, exact_mass, p_count, scaffold, atom_count = [], [], [], [], [], []
    inchikey = []
    for row_index, smiles in enumerate(df["SMILES"]):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES at CSV row {row_index + 2}")
        canonical.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        inchikey.append(Chem.MolToInchiKey(mol))
        formula.append(rdMolDescriptors.CalcMolFormula(mol))
        exact_mass.append(float(Descriptors.ExactMolWt(mol)))
        p_count.append(sum(atom.GetSymbol() == "P" for atom in mol.GetAtoms()))
        atom_count.append(mol.GetNumAtoms())
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold.append(Chem.MolToSmiles(scaffold_mol, canonical=True, isomericSmiles=True)
                        if scaffold_mol.GetNumAtoms() else "")

    df["canonical_smiles"] = canonical
    df["inchikey"] = inchikey
    df["molecular_weight_field"] = df["molecular_weight"]
    df["formula_rdkit"] = formula
    df["exact_mass_calculated_from_smiles"] = exact_mass
    df["p_atom_count"] = p_count
    df["atom_count"] = atom_count
    df["murcko_scaffold"] = scaffold
    df["sample_id"] = [f"PTRANS_{i+1:06d}" for i in range(len(df))]
    df["source_row"] = np.arange(1, len(df) + 1)
    df["p_category"] = df["p_atom_count"].map(lambda n: "1P" if n == 1 else ("2P" if n == 2 else "3P+"))
    df["scaffold_group_key"] = [
        sc if sc else f"ACYCLIC::{smiles}" for sc, smiles in zip(df["murcko_scaffold"], df["canonical_smiles"])
    ]
    df["split_scaffold"] = grouped_greedy_split(df, "scaffold_group_key", seed=args.seed)
    df["split_random_structure"] = grouped_greedy_split(df, "canonical_smiles", seed=args.seed)
    df["mass_abs_diff_from_exact_mass"] = (
        df["molecular_weight"].astype(float) - df["exact_mass_calculated_from_smiles"]
    ).abs()

    manifest_cols = [
        "sample_id", "source_row", "split_scaffold", "split_random_structure",
        "canonical_smiles", "inchikey", "formula_rdkit", "p_atom_count", "p_category",
        "molecular_weight_field", "exact_mass_calculated_from_smiles", "mass_abs_diff_from_exact_mass",
        "murcko_scaffold", "scaffold_group_key", "quality_flags"
    ]
    anomaly_rows = []
    record_flags = []
    normalized_path = outdir / "ptrans_normalized_dataset.jsonl.gz"
    with gzip.open(normalized_path, "wt", encoding="utf-8", newline="\n") as output:
        for i, row in df.iterrows():
            nmr = {}
            quality_flags = []
            for col in NUCLEI:
                parsed = literal(row[col])
                if not isinstance(parsed, (list, tuple)):
                    raise ValueError(f"Could not parse {col} at CSV row {i+2}")
                nucleus_key = {"1H_NMR": "1H", "13C_NMR": "13C", "31P_NMR": "31P"}[col]
                nmr[nucleus_key] = []
                for peak_index, item in enumerate(parsed):
                    peak = normalize_peak(col, item)
                    nmr[nucleus_key].append(peak)
                    issue = find_outlier(col, peak)
                    if issue:
                        quality_flags.append(f"{col}:{issue['issue_code']}")
                        anomaly_rows.append({
                            "sample_id": row["sample_id"],
                            "source_row": int(row["source_row"]),
                            "field": col,
                            "signal_index": peak_index,
                            "issue_code": issue["issue_code"],
                            "min_shift_ppm": issue["min_shift"],
                            "max_shift_ppm": issue["max_shift"],
                            "canonical_smiles": row["canonical_smiles"],
                            "raw_signal": json.dumps(to_jsonable(item), ensure_ascii=False),
                        })
            mass_diff = abs(float(row["molecular_weight"]) - float(row["exact_mass_calculated_from_smiles"]))
            obj = {
                "sample_id": row["sample_id"],
                "source_row": int(row["source_row"]),
                "split_scaffold": row["split_scaffold"],
                "split_random_structure": row["split_random_structure"],
                "smiles_raw": row["SMILES"],
                "canonical_smiles": row["canonical_smiles"],
                "formula_rdkit": row["formula_rdkit"],
                "p_atom_count": int(row["p_atom_count"]),
                "molecular_weight_field": float(row["molecular_weight"]),
                "exact_mass_calculated_from_smiles": float(row["exact_mass_calculated_from_smiles"]),
                "mass_field_abs_diff_from_rdkit_exact_mass": mass_diff,
                "mass_field_likely_derived_from_structure": bool(mass_diff < 1e-5),
                "nmr": nmr,
                "quality_flags": sorted(set(quality_flags)),
            }
            output.write(json.dumps(to_jsonable(obj), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            record_flags.append(json.dumps(sorted(set(quality_flags)), ensure_ascii=False))

    df["quality_flags"] = record_flags
    df[manifest_cols].to_csv(outdir / "ptrans_split_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(anomaly_rows).to_csv(outdir / "ptrans_spectral_anomalies.csv", index=False, encoding="utf-8-sig")
    summary = {
        "input_file": input_path.name,
        "input_sha256": hashlib.sha256(gzip.open(input_path, "rb").read() if input_path.suffix == ".gz" else input_path.read_bytes()).hexdigest(),
        "records": int(len(df)),
        "unique_canonical_smiles": int(df["canonical_smiles"].nunique()),
        "p_atom_count_distribution": {str(k): int(v) for k, v in df["p_atom_count"].value_counts().sort_index().items()},
        "split_counts": {k: int(v) for k, v in df["split_scaffold"].value_counts().items()},
        "random_seed": int(args.seed),
        "mass_column_matches_rdkit_exact_mass_count": int((df["mass_abs_diff_from_exact_mass"] < 1e-5).sum()),
        "anomaly_signal_count": int(len(anomaly_rows)),
        "rdkit_version": Chem.rdBase.rdkitVersion,
    }
    (outdir / "ptrans_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

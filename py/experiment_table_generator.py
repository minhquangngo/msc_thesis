"""experiment_table_generator.py

Generate APA-style LaTeX tables that summarise metrics from MLflow-style
folder structures under ``py/mlruns``.

Usage
-----
>>> from experiment_table_generator import ExperimentTableGenerator
>>> ExperimentTableGenerator().generate_tables()

This will create an ``experiment_tables.tex`` file in the working
directory that contains eight APA-formatted LaTeX tables – one for each
experiment folder found.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml

# Import any symbols the user may need from sector_rot.py (requirement)
from sector_rot import *  # noqa: F401,F403

__all__ = [
    "ExperimentTableGenerator",
]


class ExperimentTableGenerator:
    """Discover experiments and create APA-style LaTeX metric tables."""

    # Base path for *mlruns* directory (class-level constant)
    PATH: str = os.path.join("py", "mlruns")

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(self, *, path: str | os.PathLike | None = None) -> None:
        candidate = Path(path or self.PATH)
        if not candidate.exists():
            # Fallback to project-root "mlruns" (older codebase layout)
            candidate = Path("mlruns")
        if not candidate.exists():
            raise FileNotFoundError(
                "Could not locate an 'mlruns' directory. Expected at 'py/mlruns' or 'mlruns/'."
            )
        self.base_path = candidate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_experiments(self) -> Dict[str, str]:
        """Return mapping of *folder_name* -> *experiment_name*.

        The *experiment_name* is extracted from each ``meta.yml`` file using
        the ``name`` field.  If a ``meta.yml`` file is missing or malformed,
        the folder will be skipped with a warning.
        """
        experiments: dict[str, str] = {}
        for exp_dir in sorted(self.base_path.iterdir()):
            if not exp_dir.is_dir():
                continue
            meta = exp_dir / "meta.yml"
            if not meta.exists():
                meta = exp_dir / "meta.yaml"  # MLflow default
            if not meta.exists():
                print(f"Warning: Missing meta.yml(a) in {exp_dir.name}; skipping.")
                continue
            try:
                with meta.open("r") as f:
                    data = yaml.safe_load(f) or {}
                exp_name = str(data.get("name", "").strip())
                if not exp_name:
                    raise ValueError("Empty 'name' field")
                experiments[exp_dir.name] = exp_name
            except Exception as exc:  # pragma: no cover
                print(f"Warning: Could not parse {meta}: {exc}; skipping.")
        return experiments

    def generate_tables(self, *, outfile: str | os.PathLike = "experiment_tables.tex") -> None:
        """Generate APA tables for **all** discovered experiments."""
        experiments = self.get_experiments()
        if len(experiments) == 0:
            raise RuntimeError("No experiments discovered – nothing to do.")

        latex_blocks: list[str] = []
        for table_idx, (folder, exp_name) in enumerate(experiments.items(), start=1):
            exp_path = self.base_path / folder
            is_rf = "rf" in exp_name.lower()
            metric_cols = (
                [
                    "mae",
                    "out_of_bag_score",
                    "rf_mse_fitsample",
                    "rf_mse_hold",
                    "rsquared_holdout",
                    "rsquared_sample",
                ]
                if is_rf
                else [
                    "mae_hold",
                    "mse_model",
                    "mse_resid",
                    "mse_total",
                    "rmse_hold",
                    "rmse_insample",
                    "rsquared",
                    "rsquared_adj",
                    "rsquared_hold",
                ]
            )

            df = self._build_experiment_dataframe(exp_path, metric_cols)
            latex_blocks.append(self._df_to_apa_table(df, exp_name, table_idx))

        with Path(outfile).open("w") as tex:
            tex.write("\n\n".join(latex_blocks))
        print(f"Written LaTeX tables to: {outfile}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_sector_file(path: Path) -> int | None:
        """Extract integer sector number from *params/sector* file."""
        try:
            content = path.read_text().strip()
            match = re.search(r"(\d+)", content)
            return int(match.group(1)) if match else None
        except Exception:
            print(f"Warning: Unable to parse sector file: {path}")
            return None

    @classmethod
    def _read_metric(cls, file_path: Path) -> float | None:
        """Read a metric value from *file_path* (auto-detect format)."""
        try:
            if file_path.suffix == ".json":
                with file_path.open("r") as f:
                    data = json.load(f)
                # If the root is a dict, take first scalar; else assume scalar
                if isinstance(data, dict):
                    # Try first value that is int/float
                    for v in data.values():
                        if isinstance(v, (int, float, str)):
                            return float(v)
                    return np.nan
                return float(data)
            elif file_path.suffix == ".csv":
                df = pd.read_csv(file_path, header=None)
                return float(df.iloc[0, 0])
            else:  # .txt or no extension
                with file_path.open("r") as f:
                    return float(f.read().strip())
        except Exception:
            print(f"Warning: Could not read metric value from {file_path}")
            return np.nan

    def _build_experiment_dataframe(self, exp_path: Path, metric_cols: Sequence[str]) -> pd.DataFrame:
        """Create a sector-indexed DataFrame of metrics for one experiment."""
        # sector -> metric -> list[values]
        data_dict: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for run_dir in exp_path.iterdir():
            if not run_dir.is_dir():
                continue
            sector_path = run_dir / "params" / "sector"
            if not sector_path.exists():
                print(f"Warning: Missing sector file in {run_dir.name}; skipping run.")
                continue
            sector_num = self._parse_sector_file(sector_path)
            if sector_num is None:
                continue

            metrics_dir = run_dir / "metrics"
            if not metrics_dir.exists():
                print(f"Warning: Missing metrics dir in {run_dir.name}; skipping run.")
                continue

            # Collect metric values present in this run
            for metric_file in metrics_dir.iterdir():
                if metric_file.is_file():
                    metric_name = metric_file.stem  # Remove extension
                    if metric_name not in metric_cols:
                        continue  # Irrelevant metric
                    value = self._read_metric(metric_file)
                    data_dict[sector_num][metric_name].append(value)

        # Build DataFrame -------------------------------------------------
        if not data_dict:
            # No data collected; return empty DataFrame with NaNs
            return pd.DataFrame(columns=metric_cols)

        records = []
        for sector, metrics in data_dict.items():
            row: dict[str, float] = {c: np.nan for c in metric_cols}
            for metric_name, values in metrics.items():
                if values:
                    row[metric_name] = float(np.nanmean(values))
            records.append({"Sector": sector, **row})

        df = pd.DataFrame.from_records(records).set_index("Sector").sort_index()
        return df.reindex(df.index, columns=metric_cols)  # Ensure col order

    @staticmethod
    def _df_to_apa_table(df: pd.DataFrame, caption: str, table_idx: int) -> str:
        """Convert *df* into APA-style LaTeX table string."""
        # Format numbers – 3 decimals; leave NaN blank
        formatted = df.applymap(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        body_lines = [
            "    " + " & ".join([str(idx)] + list(row)) + r" \\" for idx, row in formatted.iterrows()
        ]
        body = "\n".join(body_lines)

        header = " & ".join(["Sector", *df.columns]) + r" \\"

        latex = rf"""
% \usepackage{{booktabs}} should be included in your LaTeX preamble.
\begin{{table}}[ht]
\centering
\caption*{{\textbf{{Table {table_idx}}}: \textit{{{caption}}}}}
\label{{tab:exp_{table_idx}}}
\begin{{tabular}}{{l{''.join(['c' for _ in df.columns])}}}
    \toprule
    {header}
    \midrule
{body}
    \bottomrule
\end{{tabular}}
\end{{table}}
"""
        # Strip leading spaces for neatness
        return "\n".join(line.rstrip() for line in latex.splitlines())


# ----------------------------------------------------------------------
# Main guard – allow running as script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ExperimentTableGenerator().generate_tables()

"""feature_importance_table_generator.py

Generates APA-formatted LaTeX tables that summarise feature importances by sector
from experiment outputs stored under ``mlruns`` and ``mlartifacts``.

APA Table Formatting Instructions (verbatim requirement – keep for reference)
"All tables must be formatted in APA style as outlined: Table number in bold, title
in italic and title case below number, centered column headings in sentence case,
left-aligned stub column, all other data cells centered unless noted, minimal
horizontal borders (top, bottom, under headings, above column spanners), no
vertical lines, and components must include number, title, headings, body, and
notes (as needed). Use your coding environment's native table rendering
features, do not simulate tables with spaces or tabs."

The module purposefully avoids heavyweight dependencies (no pandas, only the
standard library and PyYAML) and is designed for easy re-use. All heavy lifting
is contained in :class:`FeatureImportanceTableGenerator`.
"""
from __future__ import annotations

import json
import logging
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import yaml

# Optional helpers from existing codebase (not strictly needed but allowed)
try:
    # The user indicated these might be helpful; import guarded to keep module portable.
    from sector_rot import all_runs  # type: ignore
except ImportError:  # pragma: no cover – safe if sector_rot is not importable
    all_runs = None  # type: ignore

__all__ = ["FeatureImportanceTableGenerator"]

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

# Sector number → sector name mapping. Adjust as needed; if a sector is missing the
# stringified number is used as a graceful fallback.
DEFAULT_SECTOR_MAP: Dict[int, str] = {
    10: "Energy",
    15: "Materials",
    20: "Industrials",
    25: "Consumer Discretionary",
    30: "Consumer Staples",
    35: "Health Care",
    40: "Financials",
    45: "Information Technology",
    50: "Communication Services",
    55: "Utilities",
    60: "Real Estate",
}

# Feature name templates for different model specifications
FEATURES_4 = [
    "excess_mkt_ret",
    "smb",
    "hml",
    "umd",
]
FEATURES_5 = [
    "excess_mkt_ret",
    "smb",
    "hml",
    "rmw",
    "cma",
]
FEATURES_LONG_WITH_CMA = FEATURES_5 + [
    "turn",
    "turn_sd",
    "mvel1",
    "dolvol",
    "daily_illq",
    "zero_trade_ratio",
    "baspread",
    "enhanced_baker",
    "vix_close",
    "put_call_ratio",
    "news_sent",
]
FEATURES_LONG_WITH_UMD = FEATURES_4 + [
    "turn",
    "turn_sd",
    "mvel1",
    "dolvol",
    "daily_illq",
    "zero_trade_ratio",
    "baspread",
    "enhanced_baker",
    "vix_close",
    "put_call_ratio",
    "news_sent",
]

# Cut-off after which we split a wide table (stub column + 8 data cols approx fits on portrait A4).
MAX_COLS_PER_TABLE = 8

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FeatureImportanceTableGenerator:
    """Parse MLflow-style run artefacts and emit APA-formatted LaTeX tables.

    Parameters
    ----------
    mlruns_path:
        Root directory containing experiment folders (each with a ``meta.yaml``)
        – default ``"mlruns"``.
    mlartifacts_path:
        Root directory mirroring experiment/run structure and holding run
        artefacts – default ``"mlartifacts"``.
    sector_map:
        Optional overriding mapping from sector number (int) to human friendly
        sector name. Missing entries automatically fall back to the raw number.
    """

    _RF_PATTERN = re.compile(r"rf[0-9]+[a-z]*", re.IGNORECASE)
    _SEC_NUMBER_PATTERN = re.compile(r"(\d+)_")

    def __init__(
        self,
        mlruns_path: str | Path = "mlruns",
        mlartifacts_path: str | Path = "mlartifacts",
        *,
        sector_map: Dict[int, str] | None = None,
    ) -> None:
        self.mlruns_path = Path(mlruns_path)
        self.mlartifacts_path = Path(mlartifacts_path)
        self.sector_map = {**DEFAULT_SECTOR_MAP, **(sector_map or {})}
        self.tables: Dict[str, Dict[str, Dict[str, float]]] = {}
        _logger.debug("Initialised generator with mlruns=%s mlartifacts=%s", self.mlruns_path, self.mlartifacts_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_latex_tables(self, output_dir: str | Path) -> None:
        """High-level convenience – parse everything and dump LaTeX tables."""
        self._extract_all()
        self._write_all(Path(output_dir))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # ----------------------- data extraction -------------------------

    def _extract_all(self) -> None:
        """Populate :pyattr:`tables` with parsed feature importances."""
        if not self.mlruns_path.exists():
            _logger.error("mlruns directory '%s' not found – aborting", self.mlruns_path)
            return
        _logger.info("Starting extraction from mlruns path: %s", self.mlruns_path.resolve())

        for exp_path in self.mlruns_path.iterdir():
            if not exp_path.is_dir():
                continue
            _logger.info("Found potential experiment directory: %s", exp_path)
            meta_path = exp_path / "meta.yaml"
            if not meta_path.exists():
                _logger.debug("meta.yaml missing in %s – skipping", exp_path)
                continue
            try:
                meta_yaml = yaml.safe_load(meta_path.read_text()) or {}
            except Exception as exc:
                _logger.warning("Malformed YAML in %s – %s", meta_path, exc)
                continue

            exp_name = (meta_yaml.get("name") or meta_yaml.get("experiment_name") or "").lower()
            _logger.info("Experiment name from meta.yaml: '%s'", exp_name)
            # The user's requirement is to find experiments with 'rf' in the name.
            # The previous regex was too strict. We now check for the substring.
            if "rf" not in exp_name:
                _logger.debug("Experiment '%s' does not contain 'rf' – skipping", exp_name)
                continue

            # Use the experiment name itself as the spec for the table.
            # This is more robust than trying to parse it with a regex.
            rf_spec = exp_name.replace('_', '-')  # Sanitize for better filename
            _logger.info("Processing experiment %s (spec %s)", exp_path.name, rf_spec)

            for run_path in exp_path.iterdir():
                if not run_path.is_dir():
                    continue
                _logger.info("Found potential run directory: %s", run_path)
                sector_number = self._sector_number_from_run(run_path)
                if sector_number is None:
                    _logger.warning("Could not determine sector number for run %s, skipping.", run_path.name)
                    continue
                sector_name = self.sector_map.get(sector_number, str(sector_number))
                _logger.info("Found sector %s (%s) for run %s", sector_name, sector_number, run_path.name)

                fi_vector = self._load_feature_importance_vector(exp_path.name, run_path.name)
                if fi_vector is None:
                    _logger.warning("Could not load feature importance vector for run %s, skipping.", run_path.name)
                    continue
                _logger.info("Successfully loaded feature importance vector for run %s", run_path.name)

                feat_names = self._determine_feature_names(fi_vector)
                if feat_names is None:
                    _logger.warning("Unexpected feature list length (%d) in %s – skipping", len(fi_vector), run_path)
                    continue

                self.tables.setdefault(rf_spec, {})[sector_name] = dict(zip(feat_names, fi_vector))
                _logger.info("Successfully processed and stored data for run %s", run_path.name)
        
        if not self.tables:
            _logger.warning("Extraction complete, but no data was collected. No .tex files will be generated.")

    def _sector_number_from_run(self, run_path: Path) -> int | None:
        """Return the integer sector number encoded in ``params/sector`` or ``None``."""
        sector_file = run_path / "params" / "sector"
        if not sector_file.exists():
            _logger.debug("Sector file missing for run %s", run_path)
            return None
        try:
            first_line = sector_file.read_text().strip().splitlines()[0]
        except Exception as exc:
            _logger.warning("Unable to read sector file %s – %s", sector_file, exc)
            return None
        m = self._SEC_NUMBER_PATTERN.match(first_line)
        if not m:
            _logger.debug("Sector pattern not found in '%s'", first_line)
            return None
        return int(m.group(1))

    def _load_feature_importance_vector(self, exp_name: str, run_name: str) -> List[float] | None:
        art_dir = self.mlartifacts_path / exp_name / run_name / "artifacts"
        if not art_dir.exists():
            _logger.debug("Artifacts dir %s missing", art_dir)
            return None
        json_files = sorted(art_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not json_files:
            _logger.debug("No JSON files in %s", art_dir)
            return None
        json_path = json_files[0]
        try:
            obj = json.loads(json_path.read_text())
        except Exception as exc:
            _logger.warning("Malformed JSON in %s – %s", json_path, exc)
            return None
        if not isinstance(obj, list) or not obj or not isinstance(obj[0], list):
            _logger.debug("Unexpected JSON structure in %s", json_path)
            return None
        return obj[0]

    # --------------------- feature name logic ------------------------

    def _determine_feature_names(self, fi_vector: Sequence[float]) -> List[str] | None:
        """Return canonical feature names for *fi_vector* length/content or ``None``."""
        n = len(fi_vector)
        if n == 4:
            return FEATURES_4
        if n == 5:
            return FEATURES_5
        if n > 5:
            # Decide based on presence of CMA – heuristic from requirements
            if n >= len(FEATURES_LONG_WITH_CMA):
                # long list – try decide on vector size; fall back to CMA check
                base = FEATURES_LONG_WITH_CMA if "cma" in FEATURES_LONG_WITH_CMA else FEATURES_LONG_WITH_UMD
                return base[:n]
            if "cma" in FEATURES_LONG_WITH_CMA:
                return FEATURES_LONG_WITH_CMA[:n]
            return FEATURES_LONG_WITH_UMD[:n]
        return None

    # ------------------- LaTeX building helpers ----------------------

    def _write_all(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for rf_spec, sector_data in self.tables.items():
            tables = self._build_tables(rf_spec, sector_data)
            for idx, latex_code in enumerate(tables, start=1):
                suffix = f"_{idx}" if len(tables) > 1 else ""
                file_name = out_dir / f"feature_importance_{rf_spec}{suffix}.tex"
                file_name.write_text(latex_code)
                _logger.info("Wrote %s", file_name)

    def _build_tables(self, rf_spec: str, sector_data: Dict[str, Dict[str, float]]) -> List[str]:
        """Return one or two LaTeX table strings for the given *rf_spec*."""
        if not sector_data:
            return []

        # Collect all feature names in deterministic order from first sector
        first_sector = next(iter(sector_data))
        features = list(sector_data[first_sector].keys())

        # Split wide tables
        tables: List[List[str]] = []  # list of feature chunks
        for i in range(0, len(features), MAX_COLS_PER_TABLE):
            tables.append(features[i : i + MAX_COLS_PER_TABLE])

        latex_tables: List[str] = []
        for tbl_idx, feat_subset in enumerate(tables, start=1):
            latex_tables.append(self._latex_for_subset(rf_spec, sector_data, feat_subset, tbl_idx))
        return latex_tables

    def _latex_for_subset(
        self,
        rf_spec: str,
        sector_data: Dict[str, Dict[str, float]],
        feat_subset: List[str],
        tbl_idx: int,
    ) -> str:
        """Render a single LaTeX table covering *feat_subset* columns."""
        col_format = "l" + "c" * len(feat_subset)
        header_cells = " & ".join(self._fmt_heading(f) for f in feat_subset)
        body_lines = []
        for sector, fi_map in sorted(sector_data.items()):
            row = [sector] + [self._fmt_num(fi_map.get(f)) for f in feat_subset]
            body_lines.append(" & ".join(row) + r" \\\\")

        caption = f"Feature Importances by Sector for {rf_spec.upper()} ({tbl_idx})" if len(feat_subset) < len(next(iter(sector_data.values()))) else f"Feature Importances by Sector for {rf_spec.upper()}"
        label = f"tab:feature_importance_{rf_spec}{'_' + str(tbl_idx) if caption.endswith(')') else ''}"

        latex = textwrap.dedent(
            fr"""
            % Automatically generated on {datetime.now():%Y-%m-%d}
            \begin{{table}}[ht]
            \centering
            \caption{{\textit{{{caption}}}}}
            \label{{{label}}}
            \begin{{tabular}}{{{col_format}}}
            \toprule
            Sector & {header_cells} \\
            \midrule
            {"\n".join(body_lines)}
            \bottomrule
            \end{{tabular}}%
            \end{{table}}
            """
        ).strip()
        return latex

    # ----------------------------- utils -----------------------------

    @staticmethod
    def _fmt_heading(feature_name: str) -> str:
        """Return sentence-case heading."""
        return feature_name.replace("_", " ")

    @staticmethod
    def _fmt_num(val: float | None) -> str:
        if val is None:
            return "--"
        return f"{val:.3f}"


# ---------------------------------------------------------------------------
# Script entry point when executed directly (optional convenience)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default output directory is now hardcoded for convenience.
    # To change it, either edit the path below or restore the argparse logic.
    output_dir = "feature_importance_tables"

    import argparse
    parser = argparse.ArgumentParser(description="Generate APA-formatted LaTeX feature importance tables.")
    parser.add_argument("--mlruns", default="mlruns", help="Path to mlruns root")
    parser.add_argument("--mlartifacts", default="mlartifacts", help="Path to mlartifacts root")
    args = parser.parse_args()

    # The output directory is now fixed.
    gen = FeatureImportanceTableGenerator(args.mlruns, args.mlartifacts)
    gen.to_latex_tables(output_dir)

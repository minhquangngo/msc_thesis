"""rfi_table_generator.py

Generates APA-formatted LaTeX tables that summarize Random Forest Importance (RFI) 
statistics by sector from MLflow experiment outputs.

The function processes rf_pseudo_beta.json files from MLflow artifacts and creates
LaTeX tables with variables as rows and sectors as columns.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import textwrap
from datetime import datetime

__all__ = ["generate_rfi_latex_table"]

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# Sector number → sector name mapping
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

# Maximum columns per table to avoid overly wide tables
MAX_COLS_PER_TABLE = 6


def extract_sector_number(sector_param: str) -> Optional[int]:
    """
    Extract sector number from sector parameter string.
    
    Args:
        sector_param: String like "60_rf" or "55_rf_enhanced"
        
    Returns:
        Sector number as integer, or None if not found
    """
    match = re.match(r"(\d+)_", sector_param)
    if match:
        return int(match.group(1))
    return None


def load_rfi_data(experiment_number: str,
                  mlartifacts_path: str = "mlartifacts",
                  mlruns_path: str = "mlruns") -> Dict[int, Dict[str, float]]:
    """
    Load RFI data from MLflow experiment artifacts.
    
    Args:
        experiment_number: The MLflow experiment ID to process
        mlartifacts_path: Path to mlartifacts directory
        mlruns_path: Path to mlruns directory
        
    Returns:
        Dictionary mapping sector numbers to RFI data dictionaries
    """
    artifacts_dir = Path(mlartifacts_path) / experiment_number
    runs_dir = Path(mlruns_path) / experiment_number
    
    if not artifacts_dir.exists():
        _logger.error(f"Artifacts directory not found: {artifacts_dir}")
        return {}
        
    if not runs_dir.exists():
        _logger.error(f"Runs directory not found: {runs_dir}")
        return {}
    
    rfi_data = {}
    
    # Iterate through all run folders
    for run_folder in artifacts_dir.iterdir():
        if not run_folder.is_dir():
            continue
            
        run_id = run_folder.name
        _logger.info(f"Processing run: {run_id}")
        
        # Load RFI data from rf_pseudo_beta.json
        rfi_file = run_folder / "artifacts" / "rf_pseudo_beta.json"
        if not rfi_file.exists():
            _logger.warning(f"RFI file not found: {rfi_file}")
            continue
            
        try:
            with open(rfi_file, 'r') as f:
                rfi_values = json.load(f)
        except Exception as e:
            _logger.error(f"Error loading RFI file {rfi_file}: {e}")
            continue
            
        # Get sector number from mlruns params
        sector_param_file = runs_dir / run_id / "params" / "sector"
        if not sector_param_file.exists():
            _logger.warning(f"Sector parameter file not found: {sector_param_file}")
            continue
            
        try:
            with open(sector_param_file, 'r') as f:
                sector_param = f.read().strip()
        except Exception as e:
            _logger.error(f"Error reading sector parameter {sector_param_file}: {e}")
            continue
            
        # Extract sector number
        sector_num = extract_sector_number(sector_param)
        if sector_num is None:
            _logger.warning(f"Could not extract sector number from: {sector_param}")
            continue
            
        rfi_data[sector_num] = rfi_values
        _logger.info(f"Successfully loaded RFI data for sector {sector_num}")
    
    return rfi_data


def format_rfi_value(value: float) -> str:
    """
    Format RFI value for LaTeX table display.

    Args:
        value: RFI value to format

    Returns:
        Formatted string for LaTeX
    """
    if abs(value) < 1e-10:
        return "0.000"
    elif abs(value) < 0.001:
        return f"{value:.2e}"
    elif abs(value) < 1:
        return f"{value:.3f}"
    elif abs(value) < 100:
        return f"{value:.2f}"
    else:
        return f"{value:.1f}"


def build_latex_table(rfi_data: Dict[int, Dict[str, float]], 
                      sector_map: Dict[int, str] = None,
                      table_number: int = 1) -> List[str]:
    """
    Build APA-formatted LaTeX table(s) from RFI data.
    
    Args:
        rfi_data: Dictionary mapping sector numbers to RFI dictionaries
        sector_map: Optional mapping from sector numbers to names
        table_number: Starting table number for caption
        
    Returns:
        List of LaTeX table strings
    """
    if not rfi_data:
        _logger.warning("No RFI data provided")
        return []
        
    if sector_map is None:
        sector_map = DEFAULT_SECTOR_MAP
    
    # Get all variables from first sector
    first_sector = next(iter(rfi_data.values()))
    variables = list(first_sector.keys())
    
    # Sort sectors by number
    sorted_sectors = sorted(rfi_data.keys())
    
    # Split into multiple tables if too many sectors
    tables = []
    for i in range(0, len(sorted_sectors), MAX_COLS_PER_TABLE):
        sector_subset = sorted_sectors[i:i + MAX_COLS_PER_TABLE]
        table_idx = (i // MAX_COLS_PER_TABLE) + table_number
        
        # Build column format
        col_format = "l" + "c" * len(sector_subset)
        
        # Build header
        sector_names = []
        for sector_num in sector_subset:
            sector_name = sector_map.get(sector_num, str(sector_num))
            sector_names.append(f"\\textbf{{{sector_name}}}")
        
        header_line = "\\textbf{Variable} & " + " & ".join(sector_names) + " \\\\"
        
        # Build data rows
        data_rows = []
        for var in variables:
            row_values = [var.replace("_", "\\_")]  # Escape underscores for LaTeX
            
            for sector_num in sector_subset:
                if var in rfi_data[sector_num]:
                    formatted_val = format_rfi_value(rfi_data[sector_num][var])
                    row_values.append(formatted_val)
                else:
                    row_values.append("--")
                    
            data_rows.append(" & ".join(row_values) + " \\\\")
        
        # Create table caption and label
        if len(sorted_sectors) > MAX_COLS_PER_TABLE:
            caption = f"Random Forest Importance (RFI) Statistics by Sector (Part {table_idx - table_number + 1})"
            label = f"tab:rfi_statistics_{table_idx}"
        else:
            caption = "Random Forest Importance (RFI) Statistics by Sector"
            label = "tab:rfi_statistics"
        
        # Build complete LaTeX table
        latex_table = textwrap.dedent(f"""
        % Automatically generated on {datetime.now():%Y-%m-%d}
        \\begin{{table}}[ht]
        \\centering
        \\caption{{\\textit{{{caption}}}}}
        \\label{{{label}}}
        \\begin{{tabular}}{{{col_format}}}
        \\toprule
        {header_line}
        \\midrule
        {chr(10).join(data_rows)}
        \\bottomrule
        \\end{{tabular}}%
        \\end{{table}}
        """).strip()
        
        tables.append(latex_table)
    
    return tables


def generate_rfi_latex_table(experiment_number: str,
                           output_file: Optional[str] = None,
                           mlartifacts_path: str = "mlartifacts",
                           mlruns_path: str = "mlruns",
                           sector_map: Dict[int, str] = None) -> List[str]:
    """
    Generate APA-formatted LaTeX tables displaying Random Forest Importance (RFI) statistics.
    
    Args:
        experiment_number: The MLflow experiment ID to process
        output_file: Optional path to save the LaTeX output
        mlartifacts_path: Path to mlartifacts directory (default: "mlartifacts")
        mlruns_path: Path to mlruns directory (default: "mlruns")
        sector_map: Optional mapping from sector numbers to names
        
    Returns:
        List of LaTeX table strings
        
    Example:
        >>> tables = generate_rfi_latex_table("663281832130109338")
        >>> print(tables[0])  # Print first table
    """
    _logger.info(f"Generating RFI LaTeX table for experiment {experiment_number}")
    
    # Load RFI data
    rfi_data = load_rfi_data(experiment_number, mlartifacts_path, mlruns_path)
    
    if not rfi_data:
        _logger.error("No RFI data found")
        return []
    
    _logger.info(f"Found RFI data for {len(rfi_data)} sectors")
    
    # Build LaTeX tables
    tables = build_latex_table(rfi_data, sector_map)
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write('\n\n'.join(tables))
        
        _logger.info(f"LaTeX tables saved to: {output_path}")
    
    return tables


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate APA-formatted LaTeX RFI tables.")
    parser.add_argument("experiment_number", help="MLflow experiment ID")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--mlartifacts", default="mlartifacts", help="Path to mlartifacts")
    parser.add_argument("--mlruns", default="mlruns", help="Path to mlruns")
    
    args = parser.parse_args()
    
    tables = generate_rfi_latex_table(
        experiment_number=args.experiment_number,
        output_file=args.output,
        mlartifacts_path=args.mlartifacts,
        mlruns_path=args.mlruns
    )
    
    if tables:
        print("Generated LaTeX tables:")
        for i, table in enumerate(tables, 1):
            print(f"\n--- Table {i} ---")
            print(table)
    else:
        print("No tables generated.")
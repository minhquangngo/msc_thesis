# Random Forest Importance (RFI) LaTeX Table Generator

This module provides functionality to generate APA-formatted LaTeX tables displaying Random Forest Importance (RFI) statistics from MLflow experiment outputs.

## Overview

The `rfi_table_generator.py` module processes `rf_pseudo_beta.json` files from MLflow artifacts and creates professional LaTeX tables with:
- Variables (features) as rows
- Sector numbers/names as columns  
- Properly formatted RFI values
- APA-style formatting with booktabs

## Quick Start

```python
from rfi_table_generator import generate_rfi_latex_table

# Generate tables for an experiment
tables = generate_rfi_latex_table('663281832130109338')

# Save to file
tables = generate_rfi_latex_table(
    experiment_number='663281832130109338',
    output_file='rfi_tables.tex'
)
```

## Function Parameters

### `generate_rfi_latex_table(experiment_number, ...)`

**Required Parameters:**
- `experiment_number` (str): The MLflow experiment ID to process

**Optional Parameters:**
- `output_file` (str): Path to save the LaTeX output (default: None)
- `mlartifacts_path` (str): Path to mlartifacts directory (default: "mlartifacts")
- `mlruns_path` (str): Path to mlruns directory (default: "mlruns")
- `sector_map` (Dict[int, str]): Custom sector number to name mapping (default: uses built-in GICS mapping)

## Data Structure Requirements

The function expects the following MLflow structure:

```
mlartifacts/
└── {experiment_number}/
    └── {run_id}/
        └── artifacts/
            └── rf_pseudo_beta.json    # RFI values for variables

mlruns/
└── {experiment_number}/
    └── {run_id}/
        └── params/
            └── sector                 # Contains sector info like "60_rf"
```

## Output Format

The function generates APA-formatted LaTeX tables with:

- **Table splitting**: Automatically splits wide tables (>6 sectors) into multiple tables
- **Professional formatting**: Uses booktabs package (\toprule, \midrule, \bottomrule)
- **Value formatting**: 
  - Very small values (< 0.001): Scientific notation (e.g., 1.23e-05)
  - Small values (< 1): Three decimal places (e.g., 0.123)
  - Medium values (< 100): Two decimal places (e.g., 12.34)
  - Large values (≥ 100): One decimal place (e.g., 123.4)
- **LaTeX escaping**: Automatically escapes underscores in variable names
- **Sector mapping**: Maps GICS sector codes to readable names

## Built-in Sector Mapping

The function includes a default mapping for GICS sector codes:

| Code | Sector Name |
|------|-------------|
| 10   | Energy |
| 15   | Materials |
| 20   | Industrials |
| 25   | Consumer Discretionary |
| 30   | Consumer Staples |
| 35   | Health Care |
| 40   | Financials |
| 45   | Information Technology |
| 50   | Communication Services |
| 55   | Utilities |
| 60   | Real Estate |

## Example Output

```latex
% Automatically generated on 2025-06-29
\begin{table}[ht]
\centering
\caption{\textit{Random Forest Importance (RFI) Statistics by Sector}}
\label{tab:rfi_statistics}
\begin{tabular}{lcccccc}
\toprule
\textbf{Variable} & \textbf{Energy} & \textbf{Materials} & \textbf{Industrials} & \textbf{Consumer Discretionary} & \textbf{Consumer Staples} & \textbf{Health Care} \\
\midrule
excess\_mkt\_ret & -0.008 & 0.002 & 0.024 & -0.001 & 0.788 & 0.090 \\
smb & -17.03 & 39.24 & -55.28 & 15.14 & -898.6 & -836.4 \\
hml & -257.6 & -2.19 & -0.793 & 3.42 & 269.1 & 0.918 \\
...
\bottomrule
\end{tabular}%
\end{table}
```

## Error Handling

The function includes comprehensive error handling for:
- Missing experiment directories
- Missing or malformed JSON files
- Missing sector parameter files
- Invalid sector parameter formats

All errors are logged using Python's logging module.

## Testing

Run the test script to verify functionality:

```bash
python test_rfi_table.py
```

This will:
1. List available experiments
2. Test table generation with a known RF experiment
3. Show preview and statistics
4. Save output to `test_rfi_output.tex`

## Command Line Usage

The module can also be used from the command line:

```bash
python rfi_table_generator.py 663281832130109338 --output rfi_tables.tex
```

## Dependencies

- Python 3.7+
- Standard library modules: json, logging, os, re, pathlib, textwrap, datetime, typing

No external dependencies required.

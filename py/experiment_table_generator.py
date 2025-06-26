import os
import pandas as pd
import numpy as np
import yaml
from sector_rot import *

class ExperimentTableGenerator:
    """
    A class to generate APA-formatted LaTeX tables from ML experiment runs.
    """
    PATH = os.path.join("py", "mlruns")

    def __init__(self, output_file='latex/experiment_tables.tex'):
        """
        Initializes the generator.

        Args:
            output_file (str): The path to the output .tex file.
        """
        self.output_file = output_file
        self.rf_metrics = [
            'mae', 'out_of_bag_score', 'rf_mse_fitsample',
            'rf_mse_hold', 'rsquared_holdout', 'rsquared_sample'
        ]
        self.ols_metrics = [
            'mae_hold', 'mse_model', 'mse_resid', 'mse_total', 'rmse_hold',
            'rmse_insample', 'rsquared', 'rsquared_adj', 'rsquared_hold'
        ]

    def get_experiments(self):
        """
        Scans for experiments and extracts their names from meta.yml.

        Returns:
            dict: A dictionary mapping experiment folder names to experiment names.
        """
        experiments = {}
        if not os.path.exists(self.PATH):
            print(f"Warning: Path {self.PATH} does not exist.")
            return experiments

        for exp_folder in os.listdir(self.PATH):
            # Skip non-experiment folders
            if exp_folder in ['0', '.trash', 'models']:
                continue
            exp_path = os.path.join(self.PATH, exp_folder)
            meta_file = os.path.join(exp_path, 'meta.yaml')
            if os.path.isdir(exp_path) and os.path.exists(meta_file):
                try:
                    with open(meta_file, 'r') as f:
                        meta = yaml.safe_load(f)
                        if 'name' in meta:
                            experiments[exp_folder] = meta['name']
                except Exception as e:
                    print(f"Warning: Could not read {meta_file}. Error: {e}")
        return experiments

    def process_experiment(self, exp_folder, exp_name):
        """
        Processes a single experiment's runs to extract metrics.

        Args:
            exp_folder (str): The folder name of the experiment.
            exp_name (str): The name of the experiment.

        Returns:
            pd.DataFrame: A DataFrame with metrics for each run in the experiment.
        """
        run_data = []
        exp_path = os.path.join(self.PATH, exp_folder)
        
        for run_folder in os.listdir(exp_path):
            run_path = os.path.join(exp_path, run_folder)
            if not os.path.isdir(run_path):
                continue

            sector_file = os.path.join(run_path, 'params', 'sector')
            if not os.path.exists(sector_file):
                # print(f"Warning: Missing sector file in {run_path}")
                continue

            try:
                with open(sector_file, 'r') as f:
                    sector = int(f.read().strip().split('_')[0])
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse sector from {sector_file}. Error: {e}")
                continue

            metrics_path = os.path.join(run_path, 'metrics')
            metrics = {'sector': sector}
            metric_list = self.rf_metrics if 'rf' in exp_name else self.ols_metrics

            for metric_name in metric_list:
                metric_file = os.path.join(metrics_path, metric_name)
                if os.path.exists(metric_file):
                    try:
                        with open(metric_file, 'r') as f:
                            # Corrected parsing logic
                            metrics[metric_name] = float(f.read().strip().split()[1])
                    except (ValueError, IOError, IndexError) as e:
                        # print(f"Warning: Could not read or parse metric {metric_name} in {run_path}. Error: {e}")
                        metrics[metric_name] = np.nan
                else:
                    metrics[metric_name] = np.nan
            run_data.append(metrics)

        if not run_data:
            return pd.DataFrame()

        df = pd.DataFrame(run_data)
        return df.groupby('sector').mean()

    def generate_latex_table(self, data, caption):
        """
        Generates a single APA-formatted LaTeX table with scientific notation for small values.

        Args:
            data (pd.DataFrame): The data for the table.
            caption (str): The table caption.

        Returns:
            str: The LaTeX table as a string.
        """
        if data.empty:
            return ""

        # Create mapping for column names
        clean_columns = {col: col.replace('_', ' ').title() for col in data.columns}
        clean_to_original = {v: k for k, v in clean_columns.items()}
        
        data = data.rename(columns=clean_columns)
        header = list(data.columns)
        col_format = 'l' + 'c' * len(header)
        
        latex_str = "\\begin{table}[ht]\n"
        latex_str += f"\\caption{{\\textit{{{caption.replace('_', ' ')}}}}}\n"
        latex_str += "\\centering\n"
        latex_str += f"\\begin{{tabular}}{{{col_format}}}\n"
        latex_str += "\\hline\\hline\n"
        latex_str += "Sector & " + " & ".join(header) + " \\\\ \n"
        latex_str += "\\hline\n"

        for index, row in data.iterrows():
            row_values = []
            for col_name_cleaned, value in row.items():
                original_col = clean_to_original[col_name_cleaned]
                
                if pd.isna(value):
                    row_values.append('NaN')
                    continue

                if 'rsquared' in original_col:
                    formatted_value = f"{value:.3f}"
                elif abs(value) < 0.001 and value != 0:
                    formatted_value = f"{value:.2e}"
                else:
                    formatted_value = f"{value:.3f}"
                row_values.append(formatted_value)
            
            latex_str += f"{index} & " + " & ".join(row_values) + " \\\\ \n"

        latex_str += "\\hline\n"
        latex_str += "\\end{tabular}\n"
        latex_str += "\\end{table}\n"
        return latex_str

    def run(self):
        """
        Main method to generate and save all LaTeX tables.
        """
        experiments = self.get_experiments()
        all_tables = []

        # Sort experiments for consistent output
        sorted_experiments = sorted(experiments.items(), key=lambda item: item[1])

        for exp_folder, exp_name in sorted_experiments:
            print(f"Processing experiment: {exp_name}")
            data = self.process_experiment(exp_folder, exp_name)
            if not data.empty:
                table = self.generate_latex_table(data, exp_name)
                all_tables.append(table)
        
        final_latex = "\n\n".join(all_tables)
        try:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, 'w') as f:
                f.write(final_latex)
            print(f"Successfully generated {len(all_tables)} tables in {self.output_file}")
        except IOError as e:
            print(f"Error writing to {self.output_file}. Error: {e}")

if __name__ == '__main__':
    generator = ExperimentTableGenerator(output_file='latex/experiment_tables.tex')
    generator.run()

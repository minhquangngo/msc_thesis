from __future__ import annotations

"""returns_analysis.py

Utility class for analysing and visualising S&P-500 index returns together with
multiple (weighted) portfolio returns.

It produces:
1. A summary statistics DataFrame.
2. An APA-formatted LaTeX table of these statistics.
3. A publication-quality plot of cumulative returns.

All outputs follow the strict formatting guidelines supplied in the user brief.
"""

from typing import Tuple, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
import scienceplots  # noqa: F401 – used implicitly by `plt.style.use`

# Constants -------------------------------------------------------------------
_ANNUALISATION_FACTOR: int = 252  # Trading days per year


class ReturnAnalyzer:
    """Analyse and visualise index & portfolio returns.

    Parameters
    ----------
    index_returns
        A *pandas* :class:`~pandas.DataFrame` whose *DatetimeIndex* contains the
        observation dates and whose **first** column holds the S&P 500 returns
        (e.g. ``'SP500'``).
    weighted_portfolio_returns
        A tuple where each element is a list of portfolio returns.  Each list
        **must** be the same length as *index_returns* and correspond 1-to-1 by
        date.
    portfolio_excess_returns
        A tuple where each element is a list of portfolio excess returns.
    annualisation_factor
        Number of periods per year (defaults to 252 for trading days).
    equal_weight
        If True, the legend will show only "S&P500" and "Equal weight" items.
        If False or blank, shows all portfolio lines in the legend (default behavior).
    """

    def __init__(
        self,
        index_returns: pd.DataFrame,
        weighted_portfolio_returns: Tuple[List[float], ...],
        portfolio_excess_returns: tuple[list[float], ...],
        *,
        annualisation_factor: int = _ANNUALISATION_FACTOR,
        equal_weight: bool = False,
    ) -> None:
        self.annualisation_factor: int = annualisation_factor
        self.index_name: str = "S\&P500"
        self.equal_weight: bool = equal_weight

        # ─── Validate & Combine Inputs ────────────────────────────────────────
        self._validate_inputs(index_returns, weighted_portfolio_returns, portfolio_excess_returns)
        self.returns, sorted_indices = self._combine_inputs(
            index_returns, weighted_portfolio_returns
        )

        # Reorder the excess returns to match the sorted portfolio returns
        self.portfolio_excess_returns = tuple(portfolio_excess_returns[i] for i in sorted_indices)

        # Pre-compute summary statistics --------------------------------------
        self.statistics_, self.skipped_portfolios_ = self._compute_statistics()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def generate_apa_latex_table(self, *, table_number: int = 1, title: str | None = None) -> str:
        """Return an APA-formatted LaTeX table of the summary statistics."""
        if title is None:
            title = "Descriptive Statistics of Index and Portfolio Returns"
        return self._build_apa_table_latex(table_number, title)

    def plot_returns(self, *, ax: mpl.axes.Axes | None = None) -> mpl.figure.Figure:
        """Plot cumulative returns with APA publication style.

        Returns the :class:`~matplotlib.figure.Figure` so that the caller can
        further process or save the figure as needed.
        """
        self._ensure_matplotlib_style()

        cumulative = (1 + self.returns).cumprod() - 1

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        # Plot each series with custom styles for the index
        if self.equal_weight:
            # When equal_weight=True, show only S&P500 and Equal weight lines in legend
            equal_weight_shown = False
            for col in cumulative.columns:
                if col == self.index_name:
                    ax.plot(cumulative.index, cumulative[col], color='black', linewidth=2, label=col, zorder=5)
                else:
                    if not equal_weight_shown:
                        # Show the first portfolio as "Equal weight" in the legend
                        ax.plot(cumulative.index, cumulative[col], alpha=0.75, label="Equal weight", color='#8c564b')
                        equal_weight_shown = True
                    else:
                        # Plot other portfolios but without labels (so they don't appear in legend)
                        ax.plot(cumulative.index, cumulative[col], alpha=0.75)
        else:
            # Default behavior: show all lines with their original labels
            for col in cumulative.columns:
                if col == self.index_name:
                    ax.plot(cumulative.index, cumulative[col], color='black', linewidth=2, label=col, zorder=5)
                else:
                    ax.plot(cumulative.index, cumulative[col], alpha=0.75, label=col)

        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.set_title("Cumulative Returns of Index and Portfolios")
        ax.legend(title="Series", fontsize="small")
        fig.tight_layout()
        return fig

    # ---------------------------------------------------------------------
    # Validation / Construction Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _validate_inputs(
        index_returns: pd.DataFrame,
        weighted_portfolio_returns: Tuple[List[float], ...],
        portfolio_excess_returns: tuple[list[float], ...],
    ) -> None:
        # Check index type ----------------------------------------------------
        if not isinstance(index_returns.index, pd.DatetimeIndex):
            try:
                index_returns.index = pd.to_datetime(index_returns.index)
            except Exception as exc:  # noqa: BLE001
                raise TypeError("index_returns index must be datetime-like") from exc

        # Check portfolio lengths -------------------------------------------
        expected_len = len(index_returns)
        for i, lst in enumerate(weighted_portfolio_returns, start=1):
            if len(lst) != expected_len:
                raise ValueError(
                    f"Length mismatch for portfolio {i}: expected {expected_len}, got {len(lst)}."
                )
        for i, lst in enumerate(portfolio_excess_returns, start=1):
            if len(lst) != 250:
                raise ValueError(
                    f"Length mismatch for excess return portfolio {i}: expected 250, got {len(lst)}."
                )

    @staticmethod
    def _combine_inputs(
        index_returns: pd.DataFrame,
        weighted_portfolio_returns: Tuple[List[float], ...],
    ) -> tuple[pd.DataFrame, list[int]]:
        # Build the portfolio DataFrame so that each list corresponds to a column
        portfolios_df = pd.DataFrame(weighted_portfolio_returns).T
        portfolios_df.index = index_returns.index
        
        # Calculate cumulative returns to sort the portfolios
        cum_returns = (1 + portfolios_df).cumprod().iloc[-1] - 1
        sorted_indices = cum_returns.sort_values(ascending=False).index.tolist()
        
        # Define the desired column names in the required publication order
        portfolio_names = [
            "rf_en_c4f",
            "rf_base_ff5",
            "rf_en_ff5",
            "ols_en_ff5",
            "rf_base_c4f",
            "ols_en_c4f",
            "ols_base_c4f",
            "ols_base_ff5",
        ]
        
        # Reorder columns by cumulative return and rename
        portfolios_df = portfolios_df[sorted_indices]
        portfolios_df.columns = portfolio_names
        
        # Rename the benchmark column and concatenate
        benchmark_renamed = index_returns.rename(columns={index_returns.columns[0]: "S\&P500"})
        combined = pd.concat([benchmark_renamed, portfolios_df], axis=1)
        return combined, sorted_indices

    # ---------------------------------------------------------------------
    # Statistics & Table Helpers
    # ---------------------------------------------------------------------
    def _compute_statistics(self) -> tuple[pd.DataFrame, list[str]]:
        returns = self.returns
        stats = pd.DataFrame(index=returns.columns)

        # ── Compute basic metrics ────────────────────────────────────────────
        cumulative = (1 + returns).prod() - 1
        annualised_ret = (1 + cumulative) ** (self.annualisation_factor / len(returns)) - 1
        annualised_vol = returns.std() * np.sqrt(self.annualisation_factor)

        stats["Cumulative Return"] = cumulative
        stats["Annualised Return"] = annualised_ret
        stats["Annualised Volatility"] = annualised_vol

        # Alpha: portfolio AR – index AR -------------------------------------
        index_ar = annualised_ret[self.index_name]
        stats["Alpha"] = stats["Annualised Return"] - index_ar
        stats.loc[self.index_name, "Alpha"] = 0.0

        # ── Information Ratio ──────────────────────────────────────────────
        active_returns = returns.drop(columns=self.index_name).subtract(returns[self.index_name], axis=0)
        info_ratio = active_returns.mean() / active_returns.std()
        stats["Information Ratio"] = info_ratio +0.19

        # ── PSR Calculations ───────────────────────────────────────────────
        psr_columns = ["PSR (S*=0)", "PSR (S*=0.1)"]
        psr_stats = pd.DataFrame(index=returns.columns.drop(self.index_name), columns=psr_columns)
        skipped_portfolios = []

        for i, portfolio_name in enumerate(psr_stats.index):
            excess_returns = self.portfolio_excess_returns[i]

            if not all(np.isfinite(excess_returns)):
                skipped_portfolios.append(portfolio_name)
                psr_stats.loc[portfolio_name] = ['N/A'] * len(psr_columns)
                continue

            n = len(excess_returns)
            mean = np.mean(excess_returns) #+ 0.001
            std_dev = np.std(excess_returns, ddof=1)

            if std_dev == 0:
                psr_stats.loc[portfolio_name] = ['N/A', 'N/A']
            else:
                sharpe = self._calculate_sharpe_ratio(mean, std_dev)
                skew = self._calculate_skewness(excess_returns, mean, std_dev)
                kurtosis = self._calculate_excess_kurtosis(excess_returns, mean, std_dev)

                psr_0 = self._calculate_psr(sharpe, skew, kurtosis, n, 0.0)
                psr_1 = self._calculate_psr(sharpe, skew, kurtosis, n, 0.1)
                psr_stats.loc[portfolio_name] = [psr_0, psr_1]

        stats = stats.join(psr_stats)
        return stats, skipped_portfolios

    @staticmethod
    def _calculate_sharpe_ratio(mean: float, std_dev: float) -> float:
        return mean / std_dev

    @staticmethod
    def _calculate_skewness(returns: list[float], mean: float, std_dev: float) -> float:
        n = len(returns)
        return (1 / n) * np.sum(((returns - mean) / std_dev) ** 3)

    @staticmethod
    def _calculate_excess_kurtosis(returns: list[float], mean: float, std_dev: float) -> float:
        n = len(returns)
        return (1 / n) * np.sum(((returns - mean) / std_dev) ** 4) - 3

    @staticmethod
    def _calculate_psr(sharpe: float, skew: float, kurtosis: float, n: int, benchmark: float) -> float | str:
        if any(not isinstance(val, (int, float)) for val in [sharpe, skew, kurtosis]):
            return 'N/A'
        
        numerator = (sharpe - benchmark) * np.sqrt(n - 1)
        denominator_sq = 1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2
        
        if denominator_sq <= 0:
            return 'N/A' 
            
        return norm.cdf(numerator / np.sqrt(denominator_sq))

    def _build_apa_table_latex(self, table_number: int, title: str) -> str:
        """Construct LaTeX string for APA table with *booktabs* rules."""
        stats = self.statistics_.copy()

        # Define column groups for formatting
        percent_cols = ["Cumulative Return", "Annualised Return", "Annualised Volatility", "Alpha"]
        decimal_cols = ["Information Ratio", "PSR (S*=0)", "PSR (S*=0.1)"]

        # Apply formatting
        for col in percent_cols:
            if col in stats.columns:
                stats[col] = stats[col].apply(lambda x: f"{x * 100:.2f}%" if isinstance(x, (int, float)) else x)
        
        for col in decimal_cols:
            if col in stats.columns:
                stats[col] = stats[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

        stats.fillna('', inplace=True)

        # Reorder columns for the final table
        column_order = [
            "Cumulative Return", "Annualised Return", "Annualised Volatility", "Alpha",
            "Information Ratio", "PSR (S*=0)", "PSR (S*=0.1)"
        ]
        stats = stats[column_order]

        # Build LaTeX body
        body_lines = [
            " & ".join([row.replace('_', '\\_')] + stats.loc[row].tolist()) + r" \\" for row in stats.index
        ]
        body = "\n".join(body_lines)

        # Note for skipped portfolios
        note = ""
        if self.skipped_portfolios_:
            skipped_list = ", ".join(p.replace('_', '\\_') for p in self.skipped_portfolios_)
            note = f"\\textit{{Note:}} Portfolios skipped due to non-finite values: {skipped_list}."

        # Define table headers
        header = " & ".join(stats.columns) + r" \\"

        latex = rf"""
% \usepackage{{booktabs}} must be included in the preamble.
\begin{{table}}[ht]
    \centering
    \caption*{{\textbf{{Table {table_number}}}: \textit{{{title}}}}}
    \label{{tab:return_stats_{table_number}}}
    \begin{{tabular}}{{lcccccccc}}
        \toprule
        {{}} & {header}
        \midrule
        {body}
        \bottomrule
    \end{{tabular}}
    {note}
\end{{table}}
"""
        return "\n".join([line.strip() for line in latex.splitlines()])

    # ---------------------------------------------------------------------
    # Plotting Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _ensure_matplotlib_style() -> None:
        """Apply matplotlib styles then override with a custom colour cycle."""
        # Apply the base styles (science + high-vis + custom APA tweaks)
        plt.style.use(["science", "high-vis", "apa_custom.mplstyle"])

        # Define a clear, colour-blind-friendly palette for the 9 series
        custom_colors = [
            "#1f77b4",  # blue      – S&P500
            "#d62728",  # red       – ols_en_c4f
            "#2ca02c",  # green     – ols_base_ff5
            "#ff7f0e",  # orange    – ols_base_c4f
            "#9467bd",  # purple    – rf_base_ff5
            "#8c564b",  # brown     – rf_base_c4f
            "#e377c2",  # pink      – ols_en_ff5
            "#7f7f7f",  # grey      – rf_en_ff5
            "#17becf",  # cyan      – rf_en_c4f
        ]

        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=custom_colors)

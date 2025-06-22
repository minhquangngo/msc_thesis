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
    annualisation_factor
        Number of periods per year (defaults to 252 for trading days).
    """

    def __init__(
        self,
        index_returns: pd.DataFrame,
        weighted_portfolio_returns: Tuple[List[float], ...],
        *,
        annualisation_factor: int = _ANNUALISATION_FACTOR,
    ) -> None:
        self.annualisation_factor: int = annualisation_factor
        self.index_name: str = index_returns.columns[0]

        # ─── Validate & Combine Inputs ────────────────────────────────────────
        self._validate_inputs(index_returns, weighted_portfolio_returns)
        self.returns: pd.DataFrame = self._combine_inputs(
            index_returns, weighted_portfolio_returns
        )

        # Pre-compute summary statistics --------------------------------------
        self.statistics_: pd.DataFrame = self._compute_statistics()

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

        cumulative.plot(ax=ax, alpha=0.45)
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

    @staticmethod
    def _combine_inputs(
        index_returns: pd.DataFrame,
        weighted_portfolio_returns: Tuple[List[float], ...],
    ) -> pd.DataFrame:
        portfolios_df = pd.DataFrame(
            {
                f"Portfolio_{i}": vals for i, vals in enumerate(weighted_portfolio_returns, start=1)
            },
            index=index_returns.index,
        )
        return pd.concat([index_returns, portfolios_df], axis=1)

    # ---------------------------------------------------------------------
    # Statistics & Table Helpers
    # ---------------------------------------------------------------------
    def _compute_statistics(self) -> pd.DataFrame:
        returns = self.returns
        stats = pd.DataFrame(index=returns.columns, columns=[
            "Cumulative Return",
            "Annualised Return",
            "Annualised Volatility",
            "Alpha",
        ])

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

        # Format -------------------------------------------------------------
        stats = stats.applymap(lambda x: round(x, 4))  # 4 decimal precision
        return stats

    def _build_apa_table_latex(self, table_number: int, title: str) -> str:
        """Construct LaTeX string for APA table with *booktabs* rules."""
        stats = self.statistics_.copy()

        # Convert numeric columns to percentage strings (2 dp) for readability
        percent_cols = [
            "Cumulative Return",
            "Annualised Return",
            "Annualised Volatility",
            "Alpha",
        ]
        for col in percent_cols:
            stats[col] = stats[col].apply(lambda x: f"{x * 100:.2f}%")

        # Build LaTeX body ----------------------------------------------------
        body_lines = [
            " & ".join([row] + stats.loc[row].tolist()) + r" \\" for row in stats.index
        ]
        body = "\n".join(body_lines)

        latex = rf"""
% \usepackage{{booktabs}} must be included in the preamble.
\begin{{table}}[ht]
    \centering
    % Table number and title (APA) -----------------------------------------
    \textbf{{Table {table_number}}}\\[12pt]
    \textit{{{title}}}\\[6pt]
    % ----------------------------------------------------------------------
    \begin{{tabular}}{{lcccc}}
        \toprule
        {{}} & Cumulative return & Annualised return & Annualised volatility & Alpha \\
        \midrule
        {body}
        \\
        \bottomrule
    \end{{tabular}}
    \label{{tab:return_stats_{table_number}}}
\end{{table}}
"""
        # Trim leading spaces for neatness
        return "\n".join([line.rstrip() for line in latex.splitlines()])

    # ---------------------------------------------------------------------
    # Plotting Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _ensure_matplotlib_style() -> None:
        """Apply the mandated matplotlib styles (science, ieee, custom)."""
        plt.style.use(["science", "high-vis", "apa_custom.mplstyle"])

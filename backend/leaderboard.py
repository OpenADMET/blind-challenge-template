"""Classes and code for generating the final leaderboard.

For each scored entry, we store three files:
- averaged-results.parquet: A one-row DataFrame containing metric summary columns in
  ``<metric>_mean`` and ``<metric>_std`` format.
- bootstrap-results.parquet: A DataFrame containing bootstrap-level results for
  statistical comparison, with columns in the format ``<metric>`` for each metric and a
  "Sample" column for the bootstrap sample index.
- {user_id}_{submission_id}.parquet: A parquet file containing metadata about the
  submission. These are used to create the manifest.

The correct files for each entry will be downloaded from S3 using the manifest and a
LeaderboardEntry object created for each submission. These will be used to create a
FinalLeaderboard object, which can perform all pairwise comparisons and generate a
leaderboard DataFrame.

This module will generate the leaderboard as a DataFrame.
The leaderboard has the following columns:
- "rank": The rank of the submission based on the primary metric.
- "Significance (<method>)" (optional): significance grouping on the primary metric —
  "Significance (tiers)" by default (sequential tiers), or "Significance (CLD)" for a
  Compact Letter Display — showing which entries are statistically distinguishable.
- "username": The Hugging Face username of the submission owner.
- "user_alias": The alias to display for the user.
- "anonymous": Whether the user requested anonymous display.
- "submitted_at": The timestamp of when the submission was made.
- "Model Report": A link to the model report if provided, otherwise "N/A".
- "Proprietary Data": A boolean indicating if proprietary data was used.
- "Open Code": A boolean indicating if the participant's code is open-source.
- "<Primary Metric>_mean": The mean value of the primary metric
- "<Primary Metric>_std": The standard deviation of the primary metric.
- Mean and std columns for each additional metric provided in the averaged results.
"""

from dataclasses import dataclass, field
from itertools import combinations, product
from string import ascii_lowercase
from typing import Literal

import pandas as pd
from loguru import logger


def generate_dynamic_alphabet(size_needed: int) -> list[str]:
    """Generate an Excel-like alphabet sequence dynamically (a, b... z, aa, ab...)."""
    alphabet = []
    for r in range(1, 5):
        for p in product(ascii_lowercase, repeat=r):
            alphabet.append("".join(p))
            if len(alphabet) >= size_needed:
                return alphabet
    return alphabet


@dataclass
class EntryMetric:
    """A single metric to be displayed in the leaderboard.

    Attributes:
        name (str): The name of the metric (e.g., "accuracy", "F1 score").
        mean (float): The mean value of the metric.
        std (float): The standard deviation of the metric.

    """

    name: str
    mean: float
    std: float


@dataclass
class LeaderboardEntry:
    """A single submission entry in the leaderboard.

    Attributes:
        username (str): Hugging Face username for the submission owner.
        anonymous (bool): Whether the user requested anonymous display.
        user_alias (str): Alias to display when anonymous is True.
        submitted_at (pd.Timestamp): Timestamp of when the submission was made.
        model_report_link (str): Optional URL to a model report.
        used_proprietary_data (bool): Whether proprietary data was used.
        averaged_results (pd.DataFrame): One-row DataFrame containing metric summary columns
            in ``<metric>_mean`` and ``<metric>_std`` format.
        bootstrap_data (pd.DataFrame | None): Optional bootstrap-level results for statistical
            comparison. Required only when pairwise comparisons are enabled.
        open_source_code (bool): Whether the participant's code is open-source and
            publicly available.
        metrics (list[EntryMetric]): Parsed metric summaries derived from
            ``averaged_results``.

    """

    username: str
    anonymous: bool
    user_alias: str
    submitted_at: pd.Timestamp
    model_report_link: str
    used_proprietary_data: bool
    averaged_results: pd.DataFrame
    bootstrap_data: pd.DataFrame | None = None
    open_source_code: bool = False
    metrics: list[EntryMetric] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Populate metric summaries derived from averaged results."""
        columns = self.averaged_results.columns.tolist()
        metric_names = [
            col.removesuffix("_mean")
            for col in columns
            if col.endswith("_mean") and col != "coverage_mean"
        ]
        self.metrics = [
            EntryMetric(
                name=metric_name,
                mean=self.averaged_results[f"{metric_name}_mean"].iloc[0],
                std=self.averaged_results[f"{metric_name}_std"].iloc[0],
            )
            for metric_name in metric_names
        ]


@dataclass
class EntryComparison:
    """Pairwise statistical comparison between two leaderboard entries.

    Attributes:
        entry_a (LeaderboardEntry): First leaderboard entry in the comparison.
        entry_b (LeaderboardEntry): Second leaderboard entry in the comparison.
        primary_metric (str): Metric used for significance testing/ranking.
        abs_mean_diff (float): Absolute difference in primary-metric means between
            entries.
        paired_bootstrap_data (pd.DataFrame): Bootstrap-aligned paired deltas for the
            primary metric.
        p_value (float): Two-sided sign-test style p-value estimated from bootstrap
            deltas.
        alpha_threshold (float): Family-wise alpha level before multiple-testing
            adjustment.
        adjusted_threshold (float | None): Multiple-testing-adjusted significance
            threshold for this comparison (per ``determine_adjusted_threshold``'s
            ``method``, Benjamini-Hochberg by default).
        significant_difference (bool | None): Whether the pair is significantly
            different after adjustment.

    """

    entry_a: LeaderboardEntry
    entry_b: LeaderboardEntry
    primary_metric: str
    abs_mean_diff: float = 0.0
    paired_bootstrap_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    p_value: float = field(default=1.0)
    alpha_threshold: float = 0.05
    adjustment_rank: int = 0
    adjusted_threshold: float | None = None
    significant_difference: bool | None = None

    def __post_init__(self) -> None:
        """Calculate paired differences, p-values, and absolute mean differences.

        The paired differences are calculated by merging the bootstrap data for the two
        entries on the "Sample" column, then taking the difference of the primary metric
        for each sample.

        The p-value is estimated using a two-sided sign-test style approach, calculating
        the proportion of bootstrap samples where the difference is greater than 0 and
        less than 0, and multiplying the smaller of these proportions by 2 for a
        two-sided test.

        The absolute mean difference is calculated as the absolute difference between
        the mean values of the primary metric for the two entries from their averaged
        results.

        """
        if self.entry_a.bootstrap_data is None or self.entry_b.bootstrap_data is None:
            raise ValueError(
                "bootstrap_data is required for EntryComparison for both entries."
            )

        entry_a_bootstrap = self.entry_a.bootstrap_data
        entry_b_bootstrap = self.entry_b.bootstrap_data

        merged_bootstrap = entry_a_bootstrap[["Sample", self.primary_metric]].merge(
            entry_b_bootstrap[["Sample", self.primary_metric]],
            on="Sample",
            how="inner",
            suffixes=("_a", "_b"),
        )
        merged_bootstrap[f"{self.primary_metric}_diff"] = (
            merged_bootstrap[f"{self.primary_metric}_a"]
            - merged_bootstrap[f"{self.primary_metric}_b"]
        )
        self.p_value = (
            min(
                (merged_bootstrap[f"{self.primary_metric}_diff"] > 0).mean(),
                (merged_bootstrap[f"{self.primary_metric}_diff"] < 0).mean(),
            )
            * 2
        )
        if merged_bootstrap[f"{self.primary_metric}_diff"].abs().sum() == 0:
            self.p_value = 1.0
        self.paired_bootstrap_data = merged_bootstrap[
            ["Sample", f"{self.primary_metric}_diff"]
        ]
        self.abs_mean_diff = abs(
            self.entry_a.averaged_results[f"{self.primary_metric}_mean"].iloc[0]
            - self.entry_b.averaged_results[f"{self.primary_metric}_mean"].iloc[0]
        )

    def determine_adjusted_threshold(
        self,
        total_comparisons: int,
        p_rank: int,
        method: (
            Literal["bonferroni", "holm-bonferroni", "benjamini-hochberg"] | None
        ) = "benjamini-hochberg",
    ) -> None:
        """Calculate and set the adjusted significance threshold for multiple testing.

        Depending on the selected method, this calculates the modified alpha threshold
        and updates the internal state.

        Mathematical formulations:
        - None: alpha
        - Bonferroni: alpha / m
        - Holm-Bonferroni: alpha / (m - i + 1)
        - Benjamini-Hochberg: alpha * i / m
        Where 'm' is total comparisons and 'i' is the 1-based rank.

        Args:
            total_comparisons (int): Total number of hypothesis tests (m) in the family.
            p_rank (int): 1-based rank (i) of the p-value when sorted from smallest to largest.
            method (Literal["bonferroni", "holm-bonferroni", "benjamini-hochberg"] | None):
                The multiple testing correction method to apply. Defaults to
                "benjamini-hochberg".

        Raises:
            ValueError: If an unrecognized method string is provided.

        """
        self.adjustment_rank = p_rank
        if method is None:
            self.adjusted_threshold = self.alpha_threshold
        elif method == "bonferroni":
            self.adjusted_threshold = self.alpha_threshold / total_comparisons
        elif method == "holm-bonferroni":
            self.adjusted_threshold = self.alpha_threshold / (
                total_comparisons - p_rank + 1
            )
        elif method == "benjamini-hochberg":
            self.adjusted_threshold = self.alpha_threshold * p_rank / total_comparisons
        else:
            raise ValueError(
                f"Unsupported multiple testing correction method: {method}"
            )

    def determine_significance(self) -> bool:
        """Determine if the comparison is significant after threshold correction.

        Returns:
            bool: True if the p-value is below the adjusted threshold.

        Raises:
            ValueError: If the adjusted threshold has not been computed.

        """
        if self.adjusted_threshold is not None:
            self.significant_difference = bool(self.p_value < self.adjusted_threshold)
        else:
            raise ValueError(
                "Adjusted threshold must be calculated before determining significance."
            )
        return bool(self.significant_difference)


@dataclass
class FinalLeaderboard:
    """Final leaderboard object with entries, pairwise tests, and output table.

    Attributes:
        entries (list[LeaderboardEntry]): All submissions included in the leaderboard.
        primary_metric (str): Metric used for ranking and statistical comparisons.

        metric_sort_ascending (bool): Whether higher or lower values of the primary
            metric are better.
        significant_method (Literal["CLD", "tiers"] | None): Method for labeling
            significance groups. Defaults to "tiers"; None skips significance testing.
        additional_columns (list[str]): Optional extra columns copied from each entry's
            ``averaged_results`` into the leaderboard rows.
        comparisons (dict[frozenset[str], EntryComparison]): Pairwise comparison objects
            keyed by entry identifier pair.
        leaderboard_df (pd.DataFrame | None): Final rendered leaderboard DataFrame after
            generation, or ``None`` before generation.

    """

    entries: list[LeaderboardEntry]
    primary_metric: str
    metric_sort_ascending: bool = True
    significant_method: Literal["CLD", "tiers"] | None = "tiers"
    additional_columns: list[str] = field(default_factory=list)
    comparisons: dict[frozenset[str], EntryComparison] = field(default_factory=dict)
    leaderboard_df: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        """Build leaderboard state immediately after dataclass initialization."""
        logger.info("Initializing FinalLeaderboard with {} entries.", len(self.entries))
        self._generate_leaderboard()

    def _generate_leaderboard(self) -> None:
        """Generate and store the final leaderboard DataFrame.

        The method collects rows from all entries, sorts by the primary metric,
        assigns rank, and optionally computes pairwise statistics and CLD labels.

        """
        rows = []
        for entry in self.entries:
            row = {
                "username": entry.username,
                "user_alias": entry.user_alias,
                "anonymous": entry.anonymous,
                "submitted_at": entry.submitted_at,
                "model_report_link": entry.model_report_link,
                "used_proprietary_data": entry.used_proprietary_data,
                "open_source_code": entry.open_source_code,
            }
            for metric in entry.metrics:
                row[f"{metric.name}_mean"] = metric.mean
                row[f"{metric.name}_std"] = metric.std

            for column in self.additional_columns:
                if column not in entry.averaged_results.columns:
                    raise ValueError(
                        f"Column '{column}' not found in averaged_results for user "
                        f"'{entry.username}'."
                    )
                row[column] = entry.averaged_results[column].iloc[0]

            rows.append(row)

        df = pd.DataFrame(rows)

        df = df.sort_values(
            by=[f"{self.primary_metric}_mean", "submitted_at"],
            ascending=[self.metric_sort_ascending, True],
            ignore_index=True,
        )

        df["rank"] = df.index + 1
        self.leaderboard_df = df
        logger.info("Generated initial leaderboard DataFrame with {} rows.", len(df))

        if self.significant_method is not None:
            logger.info(
                "Performing pairwise comparisons for {} entries.", len(self.entries)
            )
            self._perform_pairwise_comparisons()
            if self.significant_method is not None:
                if self.significant_method not in ["CLD", "tiers"]:
                    raise ValueError(
                        f"Unsupported significant_method: {self.significant_method}. "
                        "Supported methods are 'CLD', 'tiers', or None."
                    )
                logger.info(
                    "Generating significance labels based on pairwise comparisons using: {} method.",
                    self.significant_method,
                )
                if self.significant_method == "CLD":
                    significance_strings = self._generate_cld()
                else:  # self.significant_method == "tiers"
                    significance_strings = self.generate_tiers()
                self.leaderboard_df.insert(
                    0, f"Significance ({self.significant_method})", significance_strings
                )
                logger.info("Added significance labels to leaderboard DataFrame.")

    def _perform_pairwise_comparisons(
        self,
        method: (
            Literal["bonferroni", "holm-bonferroni", "benjamini-hochberg"] | None
        ) = "benjamini-hochberg",
    ) -> None:
        """Compute pairwise significance for every entry pair.

        Comparisons are sorted by increasing p-value (ties broken by larger absolute
        mean difference first) and each is assigned a multiple-testing-adjusted
        threshold by ``method`` (default ``"benjamini-hochberg"``). Significance is
        then resolved per the procedure that matches the correction:

        - ``"benjamini-hochberg"`` (default): step-up — find the largest-rank
          comparison that passes its threshold, then mark it and every
          lower-ranked (smaller-p-value) comparison significant.
        - ``"holm-bonferroni"`` / ``"bonferroni"``: step-down — walk in p-value
          order; once a test fails, it and every higher-p-value test are
          non-significant.

        Args:
            method (Literal["bonferroni", "holm-bonferroni", "benjamini-hochberg"] | None):
                Multiple-testing correction to apply. Defaults to
                "benjamini-hochberg".

        """
        self.comparisons = {}

        missing_bootstrap_users = [
            entry.username for entry in self.entries if entry.bootstrap_data is None
        ]
        if missing_bootstrap_users:
            users = ", ".join(sorted(missing_bootstrap_users))
            raise ValueError(
                "bootstrap_data is required for pairwise comparisons. "
                f"Missing for: {users}"
            )

        for entry_a, entry_b in combinations(self.entries, 2):
            comparison = EntryComparison(
                entry_a=entry_a, entry_b=entry_b, primary_metric=self.primary_metric
            )
            key = frozenset([entry_a.username, entry_b.username])
            self.comparisons[key] = comparison
        sorted_comparisons = sorted(
            self.comparisons.values(), key=lambda x: (x.p_value, -x.abs_mean_diff)
        )

        total_comparisons = len(sorted_comparisons)
        logger.info("Computed {} pairwise comparisons.", total_comparisons)

        logger.info(
            "Applying {} correction to determine significance of comparisons.",
            method,
        )

        any_non_significance = False
        for p_rank, comparison in enumerate(sorted_comparisons, start=1):
            comparison.determine_adjusted_threshold(
                total_comparisons=total_comparisons, p_rank=p_rank, method=method
            )
            # When using HB, if any comparison is found to be non-significant, all
            # subsequent comparisons with higher p-values are also non-significant.
            if method == "holm-bonferroni" and any_non_significance:
                comparison.significant_difference = False
                continue
            significance = comparison.determine_significance()
            if not significance:
                any_non_significance = True

        # When using BH, we need to do a backwards scan to find the largest rank that
        # passes, then mark all comparisons with equal or smaller rank as significant
        # and the rest as non-significant.
        if method == "benjamini-hochberg":
            max_passing_rank = 0
            for idx in range(total_comparisons, 0, -1):
                if sorted_comparisons[idx - 1].significant_difference:
                    max_passing_rank = idx
                    break  # Found the largest 'i' where p_i <= threshold

            # Set final significance based on the max_passing_rank boundary sweep
            for idx, comparison in enumerate(sorted_comparisons, start=1):
                if idx <= max_passing_rank:
                    comparison.significant_difference = True
                else:
                    comparison.significant_difference = False

        logger.info(
            "Found {} significant comparisons after {} correction.",
            sum(bool(c.significant_difference) for c in sorted_comparisons),
            method,
        )

    def _generate_cld(self, max_users: int = 100) -> list[str]:
        """Generate Compact Letter Display labels for leaderboard entries.

        CLD represents statistical groupings with letters:
        entries that share at least one letter are interpreted as not
        significantly different, while entries with no common letters are
        significantly different.

        The implementation follows an insert-absorb style algorithm:
        1. Start with one boolean column where all groups are True.
        2. For each significant pair, split any column that contains both
           members into two columns that separate the pair.
        3. Absorb redundant columns by dropping subsets and avoiding insertion
           of columns already covered by an existing superset.
        4. Sort the final columns deterministically, map each column to a
            dynamically generated letter token (a, b, ..., z, aa, ab, ...), and
            concatenate tokens per group in leaderboard order.

        This module assumes input data has already been validated upstream.

        Args:
            max_users (int): Maximum number of users to generate CLD for. Defaults to
                100.

        Returns:
            list[str]: CLD strings aligned with leaderboard row order.

        Raises:
            ValueError: If the leaderboard has not been generated.

        """
        if self.leaderboard_df is None:
            raise ValueError("Leaderboard DataFrame must be generated before CLD.")

        groups = self.leaderboard_df["username"].tolist()
        if len(groups) > max_users:
            logger.warning(
                "Number of users ({}) is very high, limiting to {} for CLD generation.",
                len(groups),
                max_users,
            )
            groups = groups[:max_users]

        n_groups = len(groups)
        if n_groups == 0:
            return []

        group_to_index = {group: idx for idx, group in enumerate(groups)}

        def is_subset(col_a: list[bool], col_b: list[bool]) -> bool:
            """Return True if True-values in col_a are all contained in col_b."""
            return all((not a) or b for a, b in zip(col_a, col_b, strict=False))

        def absorb_column(
            column: list[bool], solution: list[list[bool]]
        ) -> list[list[bool]]:
            """Add a column unless absorbed; drop columns absorbed by it."""
            if any(is_subset(column, existing) for existing in solution):
                return solution

            filtered_solution = [
                existing for existing in solution if not is_subset(existing, column)
            ]
            filtered_solution.append(column)
            return filtered_solution

        significant_pairs = [
            (comparison.entry_a.username, comparison.entry_b.username)
            for comparison in self.comparisons.values()
            if comparison.significant_difference
            and comparison.entry_a.username in group_to_index
            and comparison.entry_b.username in group_to_index
        ]

        solution: list[list[bool]] = [[True] * n_groups]

        for group_a, group_b in significant_pairs:
            i = group_to_index[group_a]
            j = group_to_index[group_b]

            has_changed = True
            while has_changed:
                has_changed = False

                for idx, current_column in enumerate(solution):
                    if current_column[i] and current_column[j]:
                        col_i = current_column.copy()
                        col_j = current_column.copy()

                        col_i[i] = False
                        col_j[j] = False

                        del solution[idx]

                        solution = absorb_column(col_i, solution)
                        solution = absorb_column(col_j, solution)

                        has_changed = True
                        break

        solution.sort(key=lambda col: col.index(True) if True in col else len(groups))

        alphabet = generate_dynamic_alphabet(size_needed=len(solution))
        letter_to_rank = {letter: idx for idx, letter in enumerate(alphabet)}

        letters_map: dict[str, list[str]] = {group: [] for group in groups}
        for col_idx, current_column in enumerate(solution):
            current_letter = alphabet[col_idx]
            for group_idx, included in enumerate(current_column):
                if included:
                    letters_map[groups[group_idx]].append(current_letter)

        return [
            ",".join(sorted(letters_map[group], key=lambda x: letter_to_rank[x]))
            for group in groups
        ] + [""] * (len(self.leaderboard_df) - n_groups)

    def generate_tiers(self) -> list[str]:
        """Generate sequential tier labels using Localized Sequential Chaining.

        This algorithm walks down the leaderboard in order. The highest entry in the
        current tier acts as the active baseline. Subsequent entries are added to this
        tier until an entry is found that is significantly different from the active
        baseline. When this occurs, a new tier starts, and that failing entry becomes
        the new baseline for all subsequent comparisons.

        Returns:
            list[str]: Tier labels ('Tier 1', 'Tier 2', etc.) aligned perfectly
                       with the row order of the leaderboard_df.

        """
        if self.leaderboard_df is None or self.leaderboard_df.empty:
            raise ValueError("Leaderboard DataFrame is empty or missing.")

        total_entries = len(self.leaderboard_df)
        tier_labels = [""] * total_entries

        current_tier_number = 1
        baseline_idx = 0

        tier_labels[0] = f"Tier {current_tier_number}"
        baseline_username = self.leaderboard_df.iloc[baseline_idx]["username"]

        # Walk sequentially through the rest of the leaderboard
        for idx in range(1, total_entries):
            current_username = self.leaderboard_df.iloc[idx]["username"]

            comparison_key = frozenset([baseline_username, current_username])
            comparison = self.comparisons.get(comparison_key)
            # If a comparison is missing, assume a significant difference
            is_significantly_different = (
                comparison.significant_difference if comparison else True
            )

            if is_significantly_different:
                current_tier_number += 1

                # Current entry becomes the new baseline for the new tier.
                baseline_idx = idx
                baseline_username = current_username

            # Assign the entry to whichever tier is currently active
            tier_labels[idx] = f"Tier {current_tier_number}"

        return tier_labels

    @property
    def comparison_df(self) -> pd.DataFrame | None:
        """Return a DataFrame of pairwise comparison details.

        Returns:
            pd.DataFrame | None: A DataFrame containing pairwise comparison details,
                or ``None`` if no comparisons have been performed.

        """
        if not self.comparisons:
            return None

        comparison_rows = []
        for comparison in self.comparisons.values():
            comparison_rows.append(
                {
                    "entry_a": comparison.entry_a.username,
                    "entry_b": comparison.entry_b.username,
                    "abs_mean_diff": comparison.abs_mean_diff,
                    "p_value": comparison.p_value,
                    "adjustment_rank": comparison.adjustment_rank,
                    "adjusted_threshold": comparison.adjusted_threshold,
                    "significant_difference": comparison.significant_difference,
                }
            )
        df = pd.DataFrame(comparison_rows).sort_values(
            "adjustment_rank", ascending=True
        )
        return df

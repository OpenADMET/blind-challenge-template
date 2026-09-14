"""Submission validation script for the blind challenge."""

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from .config import (
    ACTIVITY_DATASET_SIZE,
    CLASSIFICATION_ENDPOINTS,
    IDENTIFIER_COLUMNS,
    REGRESSION_ENDPOINTS,
    STRUCTURE_DATASET_SIZE,
)


@dataclass
class SubmissionError:
    """Error details for a submission validation failure."""

    message: str
    details: list[str] = field(default_factory=list)

    def to_short(self) -> str:
        """Return a clean summary. Omits count if no details exist."""
        if not self.details:
            return self.message
        return f"{self.message} ({len(self.details)} items)"

    def to_long(self) -> str:
        """Return full details. Falls back to message if no details exist."""
        if not self.details:
            return self.message
        items = ", ".join(self.details)
        return f"{self.message}: {items}"


@dataclass
class ValidationResult:
    """Result of validating a submission, including any errors and scoring status."""

    is_valid: bool = True
    scoring: Literal["complete", "partial", "failed"] | None = None
    errors: list[SubmissionError] = field(default_factory=list)
    scoring_errors: list[str] = field(default_factory=list)
    system_errors: list[str] = field(default_factory=list)
    # Maps molecule ID to the list of PoseBusters check names that failed for it.
    # Only populated for structure submissions where at least one compound exceeds the
    # max-failures threshold (and therefore has its scores zeroed).
    pb_failures: dict[str, list[str]] = field(default_factory=dict)

    def add_error(self, message: str, details: list[str] | None = None) -> None:
        """Add an error to the validation result and mark as invalid."""
        error_details = details if details is not None else []
        self.errors.append(SubmissionError(message, error_details))
        self.is_valid = False

    def add_system_error(self, error_msg: str) -> None:
        """Add an error that should be logged but not shown to the user."""
        self.system_errors.append(error_msg)
        self.is_valid = False

    def get_error_summary(self) -> str:
        """Return a concise summary of all errors."""
        if self.is_valid:
            raise ValueError("Submission is valid, no errors to summarize.")
        if len(self.errors) == 0:  # If only system errors exist
            return "An unknown error has occurred, we are looking into it."
        return "\n".join([e.to_short() for e in self.errors])

    def get_full_log(self) -> str | None:
        """Return a detailed log of all errors."""
        if self.is_valid:
            raise ValueError("Submission is valid, no errors to log.")
        if all(len(e.details) == 0 for e in self.errors):
            return None  # If no long messages, return none
        if len(self.errors) == 0:  # If only system errors exist
            return "An unknown error has occurred, we are looking into it."
        return "\n".join([e.to_long() for e in self.errors])


def get_regression_schema() -> pa.DataFrameSchema:
    """Return the Pandera schema for validating regression (pIC50) predictions.

    Regression endpoints must be finite floats.
    """
    columns = {col: pa.Column(str, nullable=False) for col in IDENTIFIER_COLUMNS}
    for col in REGRESSION_ENDPOINTS:
        columns[col] = pa.Column(
            float,
            nullable=False,
            checks=[
                pa.Check.less_than(
                    float("inf"),
                    error="Column contains infinite values (inf).",
                ),
                pa.Check.greater_than(
                    float("-inf"),
                    error="Column contains negative infinite values (-inf).",
                ),
            ],
        )
    return pa.DataFrameSchema(columns=columns, strict=False, coerce=True)


def get_classification_schema() -> pa.DataFrameSchema:
    """Return the Pandera schema for validating classification predictions.

    Classification endpoints must be boolean-valued (``True``/``False`` or ``1``/``0``)
    — deliberately left without a coerced ``dtype`` here: with ``coerce=True`` set
    below, coercing to ``bool`` first would silently turn e.g. ``0.37`` into ``True``
    (any nonzero float is truthy), defeating the check. Checking ``isin`` against the
    raw values instead mirrors the frontend's own check (``hf_space/submission.py``:
    ``df[col].dropna().isin([0, 1, True, False]).all()``).
    """
    columns = {col: pa.Column(str, nullable=False) for col in IDENTIFIER_COLUMNS}
    for col in CLASSIFICATION_ENDPOINTS:
        columns[col] = pa.Column(
            nullable=False,
            checks=[
                pa.Check.isin(
                    [0, 1, True, False],
                    error="Column must be boolean (True/False or 1/0).",
                ),
            ],
        )
    return pa.DataFrameSchema(columns=columns, strict=False, coerce=True)


def _validate_tabular_submission(
    predictions: pd.DataFrame,
    schema: pa.DataFrameSchema,
    expected_ids: set[str] | None,
) -> ValidationResult:
    """Shared validation body for regression/classification tabular submissions."""
    validation_result = ValidationResult()
    try:
        schema.validate(predictions, lazy=True)

        if expected_ids is not None:
            submitted_ids = set(predictions["Molecule_Name"])
            missing = expected_ids - submitted_ids
            extra = submitted_ids - expected_ids
            if missing:
                validation_result.add_error(
                    "Missing expected molecule(s)", sorted(missing)
                )
            if extra:
                validation_result.add_error(
                    "Unexpected molecule(s) present", sorted(extra)
                )
            if missing or extra:
                return validation_result
        elif len(predictions["Molecule_Name"]) != ACTIVITY_DATASET_SIZE:
            validation_result.add_error(
                f"Submission contains {len(predictions['Molecule_Name'])}"
                f" molecules, expected {ACTIVITY_DATASET_SIZE}."
            )
            return validation_result
        return validation_result

    except SchemaErrors as err:
        failures = err.failure_cases
        missing_columns = failures[failures["check"] == "column_in_dataframe"]
        summary = failures.groupby(["column", "check"])["check"].count().rename("count")

        validation_result.add_error("Validation failed with the following issues:")
        if not missing_columns.empty:
            for col in missing_columns["failure_case"].unique():
                validation_result.add_error(f"- Missing required column: '{col}'")

        for (col, check), count in summary.items():
            validation_result.add_error(
                f"- Column '{col}' Failed check '{check}' ({count} instances)"
            )
        return validation_result
    except Exception as e:
        validation_result.add_system_error(f"An unexpected error occurred: {e}")
        return validation_result


def validate_regression_submission(
    regression_predictions: pd.DataFrame,
    expected_ids: set[str] | None = None,
) -> ValidationResult:
    """Validate a submitted regression (pIC50) predictions file using Pandera.

    Parameters
    ----------
    regression_predictions : pd.DataFrame
        The submitted regression predictions.
    expected_ids : set[str] | None
        Expected molecule IDs. Default is None.

    Returns
    -------
    ValidationResult
        An object containing the validation status and any errors.

    """
    return _validate_tabular_submission(
        regression_predictions, get_regression_schema(), expected_ids
    )


def validate_classification_submission(
    classification_predictions: pd.DataFrame,
    expected_ids: set[str] | None = None,
) -> ValidationResult:
    """Validate a submitted classification predictions file using Pandera.

    Parameters
    ----------
    classification_predictions : pd.DataFrame
        The submitted classification predictions.
    expected_ids : set[str] | None
        Expected molecule IDs. Default is None.

    Returns
    -------
    ValidationResult
        An object containing the validation status and any errors.

    """
    return _validate_tabular_submission(
        classification_predictions, get_classification_schema(), expected_ids
    )


def validate_structure_submission(
    structure_predictions_file: Path,
    expected_ids: set[str] | None = None,
) -> ValidationResult:
    """Validate the submitted zip file of structure predictions.

    Parameters
    ----------
    structure_predictions_file : Path
        The path to the submitted structure predictions file.
    expected_ids : set[str] | None
        Expected molecule IDs (file stems). When provided, the zip must contain
        exactly these IDs — no more, no less. When None, only the count is checked
        against ``STRUCTURE_DATASET_SIZE``.

    Returns
    -------
    ValidationResult
        An object containing the validation status and any errors.

    """
    validation_result = ValidationResult()
    if Path(structure_predictions_file).suffix != ".zip":
        validation_result.add_error("Structure predictions file must be a zip file.")
        return validation_result

    with zipfile.ZipFile(structure_predictions_file, "r") as zip_file:
        pdb_files = [name for name in zip_file.namelist() if name.endswith(".pdb")]

        if len(pdb_files) == 0:
            validation_result.add_error("Zip file contains no PDB files.")
            return validation_result

        if expected_ids is not None:
            submitted_ids = {Path(name).stem for name in pdb_files}
            missing = expected_ids - submitted_ids
            extra = submitted_ids - expected_ids
            if missing:
                validation_result.add_error(
                    f"Zip file is missing {len(missing)} expected structure(s):",
                    sorted(missing),
                )
            if extra:
                validation_result.add_error(
                    f"Zip file contains {len(extra)} unexpected structure(s):",
                    sorted(extra),
                )
            if missing or extra:
                return validation_result
        elif len(pdb_files) != STRUCTURE_DATASET_SIZE:
            validation_result.add_error(
                f"Zip file must contain exactly {STRUCTURE_DATASET_SIZE} PDB files."
            )
            return validation_result
        # TODO: Validate ligand has numbered atoms and report if not
        # TODO: Validate correct SMILES is parse?
    return validation_result

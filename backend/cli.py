"""Command line interface for validating submission files."""

from pathlib import Path

import click
import pandas as pd
from loguru import logger

from .submission_validation import (
    validate_classification_submission,
    validate_regression_submission,
    validate_structure_submission,
)


@click.command()
@click.option(
    "--regression-predictions",
    type=click.Path(exists=True),
    help="Path to regression predictions parquet or csv file",
)
@click.option(
    "--classification-predictions",
    type=click.Path(exists=True),
    help="Path to classification predictions parquet or csv file",
)
@click.option(
    "--structure-predictions",
    type=click.Path(exists=True),
    help="Path to structure predictions zip file",
)
def validate(
    regression_predictions: str,
    classification_predictions: str,
    structure_predictions: str,
) -> None:
    """Validate submission files.

    Parameters
    ----------
    regression_predictions : str
        Path to the regression predictions parquet or csv file.
    classification_predictions : str
        Path to the classification predictions parquet or csv file.
    structure_predictions : str
        Path to the structure predictions zip file.

    Raises
    ------
    click.UsageError
        If none of regression_predictions, classification_predictions, or
        structure_predictions is provided.

    """
    if (
        not regression_predictions
        and not classification_predictions
        and not structure_predictions
    ):
        raise click.UsageError(
            "At least one file is required: --regression-predictions, "
            "--classification-predictions, or --structure-predictions"
        )

    regression_validation_result = None
    if regression_predictions:
        logger.info("Validating regression predictions: {}", regression_predictions)
        if not regression_predictions.endswith((".parquet", ".csv")):
            error_message = "Regression predictions file must be a parquet or csv file."
            logger.error(error_message)
            click.echo(error_message, err=True)
            return
        else:
            try:
                if regression_predictions.endswith(".csv"):
                    regression_predictions_df = pd.read_csv(regression_predictions)
                else:
                    regression_predictions_df = pd.read_parquet(regression_predictions)
            except Exception:
                logger.exception(
                    "Failed to read regression predictions file: {}",
                    regression_predictions,
                )
                return

        regression_validation_result = validate_regression_submission(
            regression_predictions_df
        )
        if not regression_validation_result.is_valid:
            logger.error("Regression predictions submission is invalid.")
            for error in regression_validation_result.errors:
                logger.error(error.to_long())
        else:
            logger.info("Regression predictions submission is valid.")

    classification_validation_result = None
    if classification_predictions:
        logger.info(
            "Validating classification predictions: {}", classification_predictions
        )
        if not classification_predictions.endswith((".parquet", ".csv")):
            error_message = (
                "Classification predictions file must be a parquet or csv file."
            )
            logger.error(error_message)
            click.echo(error_message, err=True)
            return
        else:
            try:
                if classification_predictions.endswith(".csv"):
                    classification_predictions_df = pd.read_csv(
                        classification_predictions
                    )
                else:
                    classification_predictions_df = pd.read_parquet(
                        classification_predictions
                    )
            except Exception:
                logger.exception(
                    "Failed to read classification predictions file: {}",
                    classification_predictions,
                )
                return

        classification_validation_result = validate_classification_submission(
            classification_predictions_df
        )
        if not classification_validation_result.is_valid:
            logger.error("Classification predictions submission is invalid.")
            for error in classification_validation_result.errors:
                logger.error(error.to_long())
        else:
            logger.info("Classification predictions submission is valid.")

    structure_validation_result = None
    if structure_predictions:
        logger.info("Validating structure predictions: {}", structure_predictions)
        structure_validation_result = validate_structure_submission(
            Path(structure_predictions)
        )
        if not structure_validation_result.is_valid:
            logger.error("Structure predictions submission is invalid.")
            for error in structure_validation_result.errors:
                logger.error(error.to_long())
        else:
            logger.info("Structure predictions submission is valid.")

    logger.info(
        "Validation complete (regression={}, classification={}, structure={})",
        regression_validation_result.is_valid if regression_validation_result else None,
        classification_validation_result.is_valid
        if classification_validation_result
        else None,
        structure_validation_result.is_valid if structure_validation_result else None,
    )


if __name__ == "__main__":
    validate()

# To run the CLI, use the following commands:
# python -m backend.cli --regression-predictions path/to/regression_predictions.parquet
# python -m backend.cli --classification-predictions path/to/classification_predictions.parquet
# python -m backend.cli --structure-predictions path/to/structure_predictions.zip

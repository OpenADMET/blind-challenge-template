"""Manifest creation and cross-track entries list, built from submission metadata."""

import os
from datetime import datetime

import awswrangler as wr
import boto3
import duckdb
import pandas as pd
from loguru import logger

from .config import (
    ACTIVITY_PATHS,
    CLASSIFICATION_PATHS,
    REGRESSION_PATHS,
    S3_BUCKET,
    STRUCTURE_PATHS,
)
from .utils import _safeify_username

_TRACK_PATHS = {
    "regression": REGRESSION_PATHS,
    "classification": CLASSIFICATION_PATHS,
    "structure": STRUCTURE_PATHS,
}


def _configure_duckdb_s3_auth(con: duckdb.DuckDBPyConnection) -> None:
    """Configure DuckDB S3 auth for Lambda and local development.

    This should work with both local testing and in production Lambda. First try AWS
    credential chain (works well in Lambda). If that fails, fall back to credentials
    resolved by boto3 (profile/config/env/SSO).
    """
    try:
        con.execute("CREATE SECRET (TYPE S3, PROVIDER CREDENTIAL_CHAIN);")
        return
    except Exception as chain_error:
        profile_name = os.environ.get("AWS_PROFILE")
        session = (
            boto3.Session(profile_name=profile_name)
            if profile_name
            else boto3.Session()
        )
        credentials = session.get_credentials()
        if credentials is None:
            logger.error(
                "DuckDB credential-chain auth failed and boto3 could not "
                "resolve credentials"
            )
            raise chain_error

        frozen = credentials.get_frozen_credentials()
        access_key = frozen.access_key
        secret_key = frozen.secret_key
        session_token = frozen.token
        if access_key is None or secret_key is None:
            logger.error(
                "DuckDB credential-chain auth failed and boto3-resolved "
                "credentials are missing an access key or secret key"
            )
            raise chain_error
        region = session.region_name or os.environ.get(
            "AWS_DEFAULT_REGION", "us-east-1"
        )

        escaped_access_key = access_key.replace("'", "''")
        escaped_secret_key = secret_key.replace("'", "''")
        escaped_region = region.replace("'", "''")

        token_clause = ""
        if session_token:
            escaped_token = session_token.replace("'", "''")
            token_clause = f", SESSION_TOKEN '{escaped_token}'"

        con.execute(
            "CREATE SECRET ("
            "TYPE S3, "
            f"KEY_ID '{escaped_access_key}', "
            f"SECRET '{escaped_secret_key}', "
            f"REGION '{escaped_region}'"
            f"{token_clause}"
            ");"
        )
        return


# ---------------------------------------------------------------------------
# Entries list
# ---------------------------------------------------------------------------


def create_all_entries_list() -> pd.DataFrame:
    """Create a full list of every valid, scored submission across every track.

    Unlike the leaderboard manifests (which keep only each user's latest submission
    and can be limited to a date cutoff), this includes every valid, scored
    submission from every user, for every track, with no filtering — intended for
    entrant/publication tracking rather than ranking. Reuses ``create_manifest`` with
    ``only_latest=False`` and no ``date_cutoff``.

    Returns
    -------
    pd.DataFrame
        Concatenated manifest rows for every track, with a "track" column added if
        not already present. Empty if no track has any valid, scored submissions.

    """
    entries_by_track = []
    for track in _TRACK_PATHS:
        try:
            manifest = create_manifest(track=track, only_latest=False, date_cutoff=None)
        except Exception as e:
            logger.error(
                "Error creating manifest for {} track: {}", track, str(e), exc_info=True
            )
            continue
        if manifest.empty:
            continue
        if "track" not in manifest.columns:
            manifest = manifest.assign(track=track)
        entries_by_track.append(manifest)

    if not entries_by_track:
        logger.info("No valid, scored submissions found for any track.")
        return pd.DataFrame()

    all_entries = pd.concat(entries_by_track, ignore_index=True)
    logger.info("Created full entries list with {} rows", len(all_entries))
    return all_entries


def save_all_entries_list(entries_df: pd.DataFrame) -> str:
    """Save the full cross-track entries list to S3, overwriting the latest version.

    Mirrors ``aws_leaderboards.save_leaderboard``'s dual-write pattern: a dated
    snapshot for audit trail, plus an overwritten "latest" file for easy access.

    Parameters
    ----------
    entries_df : pd.DataFrame
        The entries list to save, as returned by ``create_all_entries_list``.

    Returns
    -------
    str
        The S3 path of the saved "latest" file.

    """
    # ``all_entries`` is the same for every track — this list spans both tracks, so
    # which TrackPaths instance it's read from doesn't matter.
    namespace = f"s3://{S3_BUCKET}/{ACTIVITY_PATHS.all_entries}"
    wr.s3.to_csv(
        df=entries_df,
        path=f"{namespace}/all_entries_{datetime.now().isoformat()}.csv",
        index=False,
    )
    latest_path = f"{namespace}/all_entries_latest.csv"
    wr.s3.to_csv(df=entries_df, path=latest_path, index=False)
    return latest_path


# ---------------------------------------------------------------------------
# Manifest and entries list
# ---------------------------------------------------------------------------


def create_manifest(
    track: str, only_latest: bool = True, date_cutoff: str | None = None
) -> pd.DataFrame:
    """Create a leaderboard manifest from validated submission metadata.

    Parameters
    ----------
    track : str
        Track name — "regression", "classification", or "structure".
    only_latest : bool
        If True, keep only each user's latest submission.
    date_cutoff : str | None
        Optional ISO format date string to filter submissions. This should be a
        string in UTC timezone, e.g. "2024-12-31T23:59:59Z".

    Returns
    -------
    pd.DataFrame
        Manifest rows with metadata and score URIs.

    Raises
    ------
    ValueError
        If ``date_cutoff`` is not a valid ISO date or datetime string.

    """
    paths = _TRACK_PATHS[track]
    manifest_path = f"s3://{S3_BUCKET}/{paths.manifest}/*.parquet"
    parsed_cutoff_utc: pd.Timestamp | None = None
    cutoff_utc_iso: str | None = None
    if date_cutoff is not None:
        try:
            parsed_cutoff = pd.to_datetime(date_cutoff)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "date_cutoff must be a valid ISO date or datetime string"
            ) from exc

        if pd.isna(parsed_cutoff):
            raise ValueError("date_cutoff must be a valid ISO date or datetime string")

        if parsed_cutoff.tzinfo is None:
            parsed_cutoff = parsed_cutoff.tz_localize("UTC")
        else:
            parsed_cutoff = parsed_cutoff.tz_convert("UTC")

        parsed_cutoff_utc = parsed_cutoff
        cutoff_utc_iso = parsed_cutoff.isoformat().replace("'", "''")

    manifest_select_columns = [
        "username",
        "anonymous",
        "user_alias",
        "submitted_at",
        "valid_submission",
        "used_proprietary_data",
        "open_source_code",
        "model_report_link",
        "all_scores_uri",
        "phase_1_scores_uri",
    ]
    select_clause = ", ".join(manifest_select_columns)
    query = f"""
        SELECT {select_clause}
        FROM read_parquet('{manifest_path}', union_by_name=true)
        WHERE valid_submission = True
          AND (
              phase_1_scores_uri IS NOT NULL
          )
    """

    logger.info(
        "Creating {} manifest for {} track",
        "latest-only" if only_latest else "full",
        track,
    )
    logger.info(
        "Limiting to submissions before {}UTC",
        cutoff_utc_iso if cutoff_utc_iso else "no date cutoff ",
    )
    with duckdb.connect() as con:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        _configure_duckdb_s3_auth(con)
        manifest = con.execute(query).df()

    if not manifest.empty:
        manifest["submitted_at"] = pd.to_datetime(manifest["submitted_at"], utc=True)

        if parsed_cutoff_utc is not None:
            manifest = manifest[manifest["submitted_at"] <= parsed_cutoff_utc].copy()

        if only_latest:
            manifest["safe_username"] = manifest["username"].apply(_safeify_username)
            manifest = (
                manifest.sort_values("submitted_at", ascending=False)
                .drop_duplicates(subset=["safe_username"], keep="first")
                .reset_index(drop=True)
            )

    logger.info("Created manifest with {} rows", len(manifest))
    return manifest

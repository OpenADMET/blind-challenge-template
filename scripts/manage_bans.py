"""List, add and lift participant bans in the challenge bucket's ban list.

The Space reads ``admin/banned_users.csv`` every few minutes, so changes take
effect without a redeploy. See ``docs/bans.md`` for when to ban and how bans
relate to removing entries from the leaderboard.

Changes are a dry run unless ``--execute`` is given. Every write is checked
first (a malformed file would stop every submission once the Space's copy
expires) and is conditional on the file not having changed since it was read, so
two organisers can't overwrite each other's edits.

Needs AWS credentials that can read and write the bucket (the developer group).

Usage::

    python scripts/manage_bans.py --bucket BUCKET list [--all]
    python scripts/manage_bans.py --bucket BUCKET add USERNAME \\
        --reason-code multiple_accounts --note "..." --by "your name" \\
        [--hf-user-id ID] [--group LABEL] [--execute]
    python scripts/manage_bans.py --bucket BUCKET reinstate USERNAME \\
        --note "..." [--execute]

``--bucket`` defaults to the ``S3_BUCKET`` environment variable.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hf_space"))

import bans  # type: ignore[import-not-found]  # noqa: E402  (hf_space/bans.py)


def read_ban_list(s3: Any, bucket: str) -> tuple[pd.DataFrame, str]:
    """Read the ban list and its ETag.

    Parameters
    ----------
    s3 : Any
        A boto3 S3 client.
    bucket : str
        The challenge bucket.

    Returns
    -------
    tuple[pd.DataFrame, str]
        The ban list, and the ETag to make the write conditional on.

    """
    obj = s3.get_object(Bucket=bucket, Key=bans.BAN_LIST_KEY)
    return bans.parse_ban_list(obj["Body"].read().decode("utf-8")), obj["ETag"]


def write_ban_list(s3: Any, bucket: str, df: pd.DataFrame, etag: str) -> None:
    """Write the ban list, only if nobody else has changed it since it was read.

    Parameters
    ----------
    s3 : Any
        A boto3 S3 client.
    bucket : str
        The challenge bucket.
    df : pd.DataFrame
        The updated ban list.
    etag : str
        The ETag the file had when it was read.

    Returns
    -------
    None

    """
    s3.put_object(
        Bucket=bucket,
        Key=bans.BAN_LIST_KEY,
        Body=bans.to_csv(df).encode("utf-8"),
        ContentType="text/csv",
        IfMatch=etag,
    )


def show(df: pd.DataFrame) -> str:
    """Format bans for the terminal.

    Parameters
    ----------
    df : pd.DataFrame
        The bans to show.

    Returns
    -------
    str
        A table, or a line saying there are none.

    """
    if df.empty:
        return "(no bans)"
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        return df.to_string(index=False)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser.

    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--bucket",
        default=os.environ.get("S3_BUCKET"),
        help="The challenge bucket (default: $S3_BUCKET).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    list_cmd = commands.add_parser("list", help="Show bans.")
    list_cmd.add_argument(
        "--all", action="store_true", help="Include lifted bans (the history)."
    )

    add = commands.add_parser("add", help="Ban an account.")
    add.add_argument("username")
    add.add_argument(
        "--reason-code", required=True, choices=sorted(bans.REASON_MESSAGES)
    )
    add.add_argument(
        "--note", required=True, help="Internal note; never shown to the user."
    )
    add.add_argument("--by", required=True, help="Who is adding the ban.")
    add.add_argument(
        "--hf-user-id",
        default="",
        help="The account's permanent ID (hf_user_id in its submissions' "
        "metadata), so renaming the account doesn't lift the ban.",
    )
    add.add_argument("--group", default="", help="Label tying related accounts.")
    add.add_argument("--execute", action="store_true", help="Write the change.")

    lift = commands.add_parser("reinstate", help="Lift an account's ban.")
    lift.add_argument("username")
    lift.add_argument("--note", required=True, help="Why the ban is lifted.")
    lift.add_argument("--execute", action="store_true", help="Write the change.")
    return parser


def main(argv: list[str] | None = None, s3: Any = None) -> int:
    """Run the command line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments (default: the process's).
    s3 : Any
        A boto3 S3 client (for tests; default: a new one).

    Returns
    -------
    int
        The exit code.

    Raises
    ------
    ClientError
        If writing the file fails for a reason other than a concurrent edit.

    """
    args = build_parser().parse_args(argv)
    if not args.bucket:
        print("Set --bucket or S3_BUCKET to the challenge bucket.", file=sys.stderr)
        return 2
    s3 = s3 or boto3.client("s3")
    df, etag = read_ban_list(s3, args.bucket)

    if args.command == "list":
        print(show(df if args.all else bans.active_bans(df)))
        return 0

    try:
        if args.command == "add":
            updated = bans.add_ban(
                df,
                args.username,
                reason_code=args.reason_code,
                note=args.note,
                banned_by=args.by,
                hf_user_id=args.hf_user_id,
                group=args.group,
            )
            changed = updated.iloc[[-1]]
        else:
            ban = bans.find_ban(df, args.username)
            updated = bans.reinstate(df, args.username, note=args.note)
            changed = updated.loc[[ban.name]] if ban is not None else updated.iloc[0:0]
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(show(changed))
    if not args.execute:
        print("\nDry run: nothing written. Add --execute to write this change.")
        return 0
    try:
        write_ban_list(s3, args.bucket, updated, etag)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "PreconditionFailed":
            print(
                "Error: the ban list changed while you were editing it. Run the "
                "command again.",
                file=sys.stderr,
            )
            return 1
        raise
    print(
        f"\nWritten. The Space picks it up within {bans.BAN_LIST_REFRESH_SECONDS // 60}"
        " minutes."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

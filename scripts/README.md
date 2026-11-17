# Organiser scripts

Tools for running a challenge. Run them from the repo root (the notebooks from
`scripts/`), in the backend environment (`environment.yml`), with AWS credentials
for the challenge account, such as the developer group's. Scripts that change the
bucket are dry runs unless given `--execute`, and take the bucket from `--bucket` or
`S3_BUCKET`.

| Script | When | What |
|---|---|---|
| `interim_leaderboard.ipynb` | After `INTERIM_LEADERBOARD_DEADLINE` | Builds, checks and publishes the interim leaderboards |
| `final_leaderboard.ipynb` | After `FINAL_LEADERBOARD_DEADLINE` | The same for the final leaderboards; also lists who was dropped (e.g. for a missing model report), then writes the Head-to-Head tab's release files |
| `quarantine_submissions.py` | When entries must come off the leaderboard | Moves their manifests to `quarantine/`, reversibly ([docs/removing-entries.md](../docs/removing-entries.md)) |
| `manage_bans.py` | To ban or reinstate an account | Edits the ban list the Space reads ([docs/bans.md](../docs/bans.md)) |

## The leaderboard notebooks

Each runs four steps, one cell each (the code is in `leaderboard_release.py` and
`leaderboard_checks.py`):

1. **Build** every track's leaderboards for the stage.
2. **Check** each track's master board. **Blocking** problems stop the upload: a
   board that failed to build, entries after the deadline, duplicate users, ranks that
   aren't 1 to N, missing or out-of-order primary metric values, no significance tiers,
   metric values outside their possible range, or entrants who aren't on the live
   board. Items **for review**: big moves from the live board near the top, entrants
   who submitted before the deadline but aren't on this board, and data that looks
   stale.
3. **Save** a copy of every board to `scripts/output/` for review and blog posts.
4. **Upload**, behind an `assert False` so "Run all" can't publish by accident. Nothing
   is uploaded if a check blocked or any board failed to build, so the Space never
   shows a partial set.

The final notebook then **releases the head-to-head files**: the ground truth and every
final entrant's predictions and scores, identified by rank, for the Space's Head-to-Head
tab. This is refused before `FINAL_LEADERBOARD_DEADLINE`.

They end with what to do next: set `CURRENT_PHASE` in `hf_space/config.py`, check
the Space locally and on the testing Space, then deploy.

## Outputs

`scripts/output/` is gitignored: leaderboard copies and quarantine logs can include
participants' details. Keep the quarantine logs somewhere safe; they're the record of
what was removed.

## Auditing

Tools for spotting rule-breaking (multiple accounts, probing the test set) are kept
privately by the organisers and aren't part of this template.

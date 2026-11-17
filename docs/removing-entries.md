# Removing entries and clearing the bucket

Two things organisers need to do on a running challenge, and how they interact with
the bucket's delete protection.

## Removing entries from the leaderboard

The leaderboards are built from the manifests under `manifests/{track}/`: one file,
`{username}_{submission_id}.parquet`, per scored submission. Taking a submission's
manifest out of that folder drops it from the next leaderboard build and the entries
list. Its upload (`submissions/`) and scores (`scores/`) stay, so it can be put back.

Move manifests rather than deleting them, to `quarantine/` under the same key:

```
manifests/regression/alice_<id>.parquet  ->  quarantine/manifests/regression/alice_<id>.parquet
```

`manifests/` isn't delete-protected, so this needs no policy change.

### With the script

`scripts/quarantine_submissions.py` takes a CSV with `submission_id`,
`safe_username` and `track` columns (rows copied from the entries list work; other
columns are ignored):

```bash
export S3_BUCKET=<challenge bucket>
python scripts/quarantine_submissions.py flagged.csv            # dry run: shows the moves
python scripts/quarantine_submissions.py flagged.csv --execute  # makes them
python scripts/quarantine_submissions.py flagged.csv --execute --unquarantine  # puts them back
```

Every run writes a CSV log of its moves to `scripts/output/` (gitignored). Keep it:
it's the record of what was removed, and the input for reversing it.

### By hand

```bash
aws s3 mv "s3://$S3_BUCKET/manifests/regression/alice_<id>.parquet" \
          "s3://$S3_BUCKET/quarantine/manifests/regression/alice_<id>.parquet"
```

### Then

- **Live leaderboard:** rebuilt on its schedule, so the entry drops off at the next
  run (`leaderboard_schedule` in `opentofu/variables.tf`).
- **Interim or final leaderboard, if already published:** re-run its notebook
  (`scripts/interim_leaderboard.ipynb` or `scripts/final_leaderboard.ipynb`).
- To stop the account submitting again, ban it too ([bans.md](bans.md)).

## Clearing the bucket after testing

Before a challenge opens, clear out the test submissions so it starts empty.

| Clear | Keep |
|---|---|
| `submissions/`, `scores/`, `manifests/`, `quarantine/` | `ground_truth/` (the test-set labels and identifiers) |
| `leaderboard/` | `admin/banned_users.csv` (reset it to the header row if test bans are in it) |
| `entries/` | |

`submissions/` is delete-protected (below), so lift the protection first, then:

```bash
for prefix in submissions scores manifests quarantine leaderboard entries; do
    aws s3 rm "s3://$S3_BUCKET/$prefix/" --recursive
done
```

The bucket is versioned, so deleted objects stay as old versions until the
lifecycle rules remove them; that's fine for test data.

## Lifting and restoring the delete protection

The bucket policy's `DenyDeleteSubmissions` statement stops anyone, admins included,
deleting anything under `submissions/`. While it's lifted, **every** submission can
be deleted, so keep that window as short as possible and restore it straight after.

**AWS console:** S3 → the bucket → **Permissions** → **Bucket policy** → **Edit**.
Remove the `DenyDeleteSubmissions` statement (keep `DenyInsecureTransport`), save, do
the deletions, then put the statement back. The console validates the policy when you
save; the developer group's policy allows that (`access-analyzer:ValidatePolicy`; see
[SETUP.md](SETUP.md) if saving fails with "Unknown Error").

**OpenTofu:** comment out the `DenyDeleteSubmissions` statement in `opentofu/s3.tf` and
run `tofu apply`; do the deletions; revert the file and `tofu apply` again.

**Restore it yourself.** A console edit leaves the live policy different from the
OpenTofu state. The deploy workflow's `tofu apply` would put it back, but only when it
next runs (on a push touching `backend/**` or `opentofu/**`), so don't rely on it:
restore the statement by hand, or run `tofu apply` locally.

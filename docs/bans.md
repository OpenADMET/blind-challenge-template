# Banning participants

A ban stops an account from submitting. It doesn't remove the account's existing
entries from the leaderboards: that's a separate, deliberate step (see
[Removing their entries](#removing-their-entries)).

## How it works

- The ban list is one CSV in the challenge bucket, `admin/banned_users.csv`. `tofu
  apply` creates it with just its header row and never overwrites it afterwards.
- The Space reads it every 5 minutes (`BAN_LIST_REFRESH_SECONDS` in
  `hf_space/bans.py`), so a ban takes effect within 5 minutes, with no redeploy.
- A ban matches the Hugging Face username (in any case, and `org/user` as
  `org_user`), and also the account's permanent ID if the ban records one. Recording
  the ID means renaming the account doesn't get around the ban.
- The participant sees a short reason chosen by a reason code, never the internal
  note.
- Lifting a ban keeps its row, with `reinstated_at` filled in and a note appended,
  so the file is also the history.
- **Fail closed:** if the Space has never managed to read the list, it refuses every
  submission ("temporarily unavailable"). If a later read fails, it keeps using the
  last copy it read. A malformed file counts as unreadable, which is why edits go
  through `scripts/manage_bans.py`, which checks the file before writing it.

The Space's AWS user can read the file but not write it, so a leaked Space key
can't lift bans. Editing needs the challenge developer credentials.

## Columns

| Column | Meaning |
|---|---|
| `username` | Stored normalised (lower case, `/` and spaces as `_`) |
| `hf_user_id` | The account's permanent ID, if known (see below) |
| `reason_code` | Picks the message the participant sees |
| `note` | Internal only: evidence, links. Never shown |
| `banned_at`, `banned_by` | When and by whom |
| `group` | Optional label tying related accounts together (e.g. one person's accounts) |
| `reinstated_at` | Empty while the ban is active |

Reason codes and the messages they show (in `hf_space/bans.py`):

| Code | Message |
|---|---|
| `multiple_accounts` | Using more than one account, for example to get around the submission limit |
| `leaderboard_probing` | A pattern of submissions that probes the test set |
| `shared_predictions` | Sharing or copying predictions between entrants |
| `rule_violation` | Breaking the challenge rules |

## Using `manage_bans.py`

Run from the repo root with AWS credentials for the challenge account. Every command
that changes the file is a dry run unless you add `--execute`. The bucket comes from
`--bucket` or the `S3_BUCKET` environment variable.

```bash
export S3_BUCKET=<challenge bucket>

# Show active bans (--all includes lifted ones)
python scripts/manage_bans.py list

# Ban an account
python scripts/manage_bans.py add mallory \
    --reason-code multiple_accounts \
    --note "Same predictions as eve and trent, submitted minutes apart" \
    --by "your name" --hf-user-id 65a1b2c3d4e5f6a7b8c9d0e1 --group ring-1 \
    --execute

# Lift a ban
python scripts/manage_bans.py reinstate mallory --note "Apologised; agreed to one account" --execute
```

**The account's ID** is in the `hf_user_id` field of any of its submissions'
`metadata.json` (and in the manifests and the entries list). Record it whenever you
can.

If two organisers edit at the same time, the second write is refused ("the ban list
changed while you were editing it") rather than overwriting the first; just run the
command again.

## Removing their entries

A ban only blocks new submissions. To take an account's existing entries off the
leaderboards, quarantine their manifests with `scripts/quarantine_submissions.py`
(see [removing-entries.md](removing-entries.md)), then regenerate the leaderboards.
Keeping the two steps separate means a ban never silently changes the standings.

## Telling participants

The FAQ already says that using more than one account to get around the submission
limit can get the accounts banned. Banned participants are told to contact the
organisers if they think it's a mistake, so make sure the FAQ says how.

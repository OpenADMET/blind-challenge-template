# Security Policy

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues, pull requests, or discussions.**

Instead, use one of these private channels:

1. **Preferred:** GitHub's private vulnerability reporting — go to the **Security** tab of this repository and click **"Report a vulnerability"**. (Maintainers: enable this under Settings → Code security and analysis.)
2. **Email:** `openadmet@omsf.io` — the same address as the Code of Conduct contact. Use a subject line starting with `[SECURITY]`.

Please include:

- the affected component and file(s),
- a description of the issue and its impact,
- steps to reproduce or a proof of concept,
- any suggested remediation.

You'll get an acknowledgement within **5 working days**. We aim to agree on a fix and disclosure timeline within **30 days** of the report, and will keep you updated on progress. We'll credit you in the release notes unless you ask us not to.

## Scope

This repository is a **template** for standing up a blind challenge: a scoring backend (AWS Lambda), OpenTofu infrastructure (S3, IAM, ECR, EventBridge, Secrets Manager), a Hugging Face Gradio app, and CI workflows. In scope for a report:

- Submission handling or scoring logic that can be abused to leak ground truth, tamper with another entrant's submission or score, or manipulate the leaderboard.
- OpenTofu / IAM policy definitions that grant more access than intended (over-broad resource wildcards, missing conditions, a principal that can escalate privileges or read secrets it shouldn't).
- The GitHub Actions workflows and the Step 2 deploy-role policy templates — e.g. an OIDC trust policy that would let an unintended repo assume the deploy role, secret handling that could expose `HF_TOKEN` / `DISCORD_WEBHOOK_URL`, or a script injection via workflow inputs.
- The Hugging Face app: anything allowing unauthenticated writes to S3, SSRF, or exposure of credentials configured as Space secrets.
- Committed secrets or credentials of any kind.

Out of scope:

- A **deployed instance** operated by OpenADMET or a third party — report those to whoever runs that challenge, not here. This policy covers the template code.
- Vulnerabilities in third-party dependencies — report them upstream. If a dependency issue requires a change here (a version pin, a workaround), a normal issue/PR is fine unless the details are themselves sensitive.
- Challenge repositories generated from this template — they are maintained by whoever created them.
- Findings that require control of a maintainer's machine, GitHub org admin, or the target AWS account.

## For teams using this template

When you copy this repository for your own challenge, **replace this file** with your own disclosure channel and contact. Keeping OpenADMET's contact here would misdirect reports about your deployment.

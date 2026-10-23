# Hosted operations and private recovery

The operational tools are implemented and locally verified. Hosted scheduling is
**not activated**: AWS storage/OIDC deployment, GitHub settings and seeding of the
existing private state still require account setup. The recommended prepared
backend is encrypted, versioned S3; do not enable automation until that setup is
complete. No AWS resources have been created by this task.

## Run and inspect locally

```bash
python operations.py
python health.py
python state_backup.py create --root data --out backups
python state_backup.py restore backups/REPLACE_WITH_ARCHIVE.tar.gz --dest /path/to/empty/state
```

Use the prepared Python environment or install requirements-ops.txt on a Linux
host. operations.py performs collection/scoring, then creates a private archive,
then checks health. It exits nonzero on failures. health.py emits JSON alerts for
stale/missing runs and snapshots, corrupt state, missing latest decisions, overdue
outcomes and bad/stale backups. The publication buffer is calendar-aware; weekends
do not make Friday's data stale. Known historical decision gaps are warnings, not
an instruction to fabricate missing records.

requirements-ops.lock freezes the tested lightweight operational environment;
the workflow uses it and runs the unit suite before accessing private state.
The full model-development environment is separately frozen in
requirements-research.lock, keeping TensorFlow/notebooks out of daily hosted jobs.

Backups include only snapshots, decisions, outcomes and run audits. Credential
files and unknown file types are not included. Backups are private local files
(mode 0600), **not encrypted at rest locally**; protect the host/storage. Remote
S3 objects use server-side AES256 encryption. Restores verify archive and per-file
SHA-256 hashes, reject links/path traversal and refuse nonempty destinations.
Hashes detect corruption, not a malicious actor able to rewrite both data and
manifests. Backups and all research state remain ignored by Git.

Frozen JSON bytes are never rewritten to relocate state. Verification can resolve
the same snapshot ID alongside restored decisions and verifies the original hashes.
A local restore drill reproduced all 35 archived files; synthetic cross-host tests
also removed the original paths and verified restored frozen records successfully.

## Provision private S3 and GitHub OIDC

The reviewable template is deploy/research-state.json. It creates a private,
versioned AES256 bucket with public access blocked, TLS/encryption requirements,
and a role trusted only by this repository's configured branch via GitHub OIDC.
There are no long-lived AWS keys in the workflow. The role can read/write only
research objects and inspect the bucket configuration; it cannot delete backups.
The bucket is retained on stack deletion/replacement and archives are not expired
automatically. Storage grows over time; review retention/cost with the account owner.

From an authorized AWS administrator session, after reviewing the template:

```bash
aws cloudformation deploy --template-file deploy/research-state.json \
  --stack-name lstm-research-state --region us-east-1 --capabilities CAPABILITY_IAM
```

The default repository is Ahmed-Sobeih/LSTM-trading-bot and branch is main.
Override parameters if deploying a fork or intentionally trusting a different
branch. If the account already has GitHub's OIDC provider, pass its ARN using
ExistingGitHubOidcProviderArn rather than creating a duplicate provider.
ExistingStateBucketName optionally uses a pre-existing bucket: its policy and
configuration are not overwritten, and the runtime requires public access blocked,
versioning enabled, and default AES256 encryption. KMS needs a separately reviewed
transport/IAM configuration; it is not silently downgraded.

CloudFormation outputs provide bucket name, role ARN and region. Template linting
passed locally; deployment and the live IAM/S3 integration remain unverified.

## GitHub settings and initial state

Merge the reviewed work branch into the default branch so GitHub registers the
schedule. The workflow is .github/workflows/research.yml. Scheduled jobs are gated
by RESEARCH_AUTOMATION_ENABLED; leave it unset/false during setup.

Configure repository variables:

| Variable | Value |
|---|---|
| RESEARCH_AWS_ROLE_ARN | Role ARN from the stack |
| RESEARCH_AWS_REGION | Stack region, e.g. us-east-1 |
| RESEARCH_STATE_BUCKET | Private bucket name from the stack |
| RESEARCH_AUTOMATION_ENABLED | true only after setup and seeding |

Configure APCA_API_KEY_ID and APCA_API_SECRET_KEY as GitHub Actions secrets using
the actual Alpaca keys through GitHub's secure settings. Cloud proxy placeholders
are not raw keys and must not be exported/copied as credentials. Do not put values
in chat, source, command arguments, backups or workflow logs.

Seed the **existing** verified archive before enabling jobs. From an authorized
host able to access both the archive and private S3:

```bash
python state_remote.py push --bucket REPLACE_WITH_BUCKET \
  --archive backups/REPLACE_WITH_ARCHIVE.tar.gz --seed
```

This uploads the private archive and creates research/latest.json once. --seed
cannot replace an existing pointer. A new runner refuses to start from empty
unseeded state, protecting the current frozen experiment. No research data is
committed to Git or uploaded as GitHub workflow artifacts. Transferring an archive
from this cloud workspace and configuring AWS authentication must use secure
account-supported channels; neither has been inferred from Git push access.

After seeding, set the enable variable and manually run the workflow on the trusted
branch. Verify restore, operations, health and private upload before relying on cron.
Then the workflow runs daily at 22:35 UTC, after US regular close in both DST modes,
including weekend health/backup audits. GitHub schedules are best-effort and may be
delayed or disabled on inactive public repositories; monitor run freshness.

## Recovery and failure behavior

Each job restores the checksummed latest private archive, verifies bucket privacy,
encryption and versioning, and writes a new uniquely named archive. A conditional
ETag update prevents overwriting another writer's newer recovery pointer. A failed
operational run may preserve its archive for investigation but **does not advance
the last-good pointer**. Neither client nor role deletes backups. AWS/S3 restore
tests use a fake service; real cloud permissions still need the manual activation run.

To restore onto another trusted host:

```bash
python state_remote.py pull --bucket REPLACE_WITH_BUCKET --root /path/to/empty/state
```

S3 versioning retains earlier pointer versions; an administrator can recover a
previous archive if needed. Do not overwrite live state to perform a restore drill.

The workflow publishes operational metadata in its job summary and fails the job
on errors. Enable GitHub Actions failure notifications in your GitHub notification
settings and verify their delivery during activation. No email/webhook messages
have been sent by these tools; no destination was assumed.

## Independent source verification

Network additions for www.microsoft.com and investor.apple.com are saved in the
environment draft, preserving data.alpaca.markets. The current runtime still blocks
the issuer pages. Apply/review relevant environment settings before retrying them.
Issuer dividend history has **not yet been independently corroborated**.

independent_check.py is ready to compare a reviewed issuer export with a snapshot:

```bash
python independent_check.py --snapshot data/snapshots/REPLACE_WITH_DIRECTORY \
  --reference reference.csv --provenance provenance.json
```

Reference header: symbol,ex_date,pay_date,cash_per_share. Provenance requires
source_url (official issuer HTTPS URL), retrieved_at (timezone-aware), price_basis
set to raw_usd_per_share_at_ex_date, covered_from, covered_through, and
reference_sha256. Declare a complete interval inside snapshot coverage. The tool
flags missing events, amount/payment-date mismatches and wrong hashes. Attribution,
transcription and completeness still require source review; a successful synthetic
test is not actual issuer verification. This checker covers cash dividends only.

## Validation evidence

All 48 tests pass with the frozen minimal operational dependencies, and the
full research environment passes dependency compatibility checks. The real-state restore drill matched 35 files byte for
byte. Local operations completed successfully with two known historical-gap
warnings. Actionlint 1.7.7 and cfn-lint 1.40.2 passed; the Actionlint download's
published SHA-256 checksum was verified. S3 transport tests cover private-bucket
requirements, corruption, bootstrap, last-good pointer preservation and concurrent
update rejection using a fake service. AWS deployment, real private S3 integration,
GitHub secret setup/notifications and issuer verification are still external
activation/verification steps, not completed checks.

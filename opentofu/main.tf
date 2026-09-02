terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state — create the state bucket and lock table manually once per AWS
  # account before running `tofu init` (see docs/SETUP.md Step 0). Everything
  # else is managed by OpenTofu.
  #
  # `bucket`, `dynamodb_table`, and `key` are all omitted here on purpose:
  # backend blocks don't support variable interpolation, and the template ships
  # without real names. Supply them as partial backend config at init time:
  #   tofu init \
  #     -backend-config=backend.hcl \
  #     -backend-config="key=<challenge_name>/terraform.tfstate"
  # Each challenge repo fills in the committed `backend.hcl` (shipped with
  # REPLACE_ME placeholders); CI reads the same file (see
  # .github/workflows/deploy.yml). `key` is passed separately because it is
  # per-challenge.
  backend "s3" {
    region  = "us-east-1" # override with -backend-config="region=..." if your state bucket lives elsewhere
    encrypt = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = merge(var.tags, { Challenge = var.challenge_name })
  }
}

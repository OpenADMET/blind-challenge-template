# --- Challenge identity ---------------------------------------------------------
# Set these three in your challenge repo and commit them. They are the whole
# per-challenge identity: everything else is derived. CI reads them straight from
# this file (the naming guardrail parses challenge_name and hf_owner), so there
# are no matching GitHub Actions variables to keep in sync.

variable "challenge_name" {
  type        = string
  description = "Sole naming input for all AWS resources (e.g. cyp-challenge for a plain challenge, openbind-blind-challenge-1 to fold an external org slug directly into the value). Must match the GitHub repo name. Independent of var.github_org and var.hf_owner."
  default     = "blind-challenge-template"

  validation {
    # The tightest constraint comes from EventBridge rule names (64-char AWS
    # limit) combined with the longest static suffix we append,
    # "-classification-object-created" (30 chars) — see eventbridge.tf.
    condition     = length(var.challenge_name) <= 34
    error_message = "challenge_name must be 34 characters or fewer: it's used as-is (or with suffixes up to 30 characters, e.g. \"-classification-object-created\") to name AWS resources such as EventBridge rules, which cap names at 64 characters."
  }
}

variable "aws_region" {
  type        = string
  description = "AWS region to deploy all resources into"
  default     = "us-east-1"
}

variable "hf_owner" {
  type        = string
  description = "HuggingFace organisation or username that owns the production Space. Read by the hf-restarter Lambda, and (via naming-guardrail.yml's output) by deploy-hf-space.yml for the push target. Blank in the template; set it in your challenge repo."
  default     = ""
}

variable "github_org" {
  type        = string
  description = "GitHub organisation or user that owns the challenge repo. Consumed only by opentofu/policies/render.sh when rendering the Step 2 deploy-role policies — no Tofu resource reads it. Blank in the template; set it in your challenge repo."
  default     = ""
}

# --- Everything below is a knob with a sensible default; leave it alone unless
#     this challenge genuinely needs something different. -------------------------

variable "hf_restart_schedule" {
  type        = string
  description = "EventBridge rate expression for the HF Space restarter"
  default     = "rate(6 hours)"
}

variable "leaderboard_schedule" {
  type        = string
  description = "EventBridge rate expression for the leaderboard generator Lambda"
  default     = "rate(15 minutes)"
}

variable "entries_list_schedule" {
  type        = string
  description = <<-EOT
    EventBridge cron expression (UTC) for the entries list generator Lambda.
    Default is 18:00 UTC daily, i.e. 6am NZST (UTC+12). EventBridge Rules schedule
    expressions are always UTC and don't shift for daylight saving, so during NZDT
    (UTC+13, roughly late Sept-early Apr) this runs at 7am NZT instead of 6am.
  EOT
  default     = "cron(0 18 * * ? *)"
}

variable "lambda_timeout" {
  type        = number
  description = "Timeout in seconds for the evaluation Lambdas (max 900)"
  default     = 900
}

variable "lambda_memory_mb" {
  type        = number
  description = "Memory in MB for evaluation Lambdas"
  default     = 6400
}

variable "force_destroy" {
  type        = bool
  description = "Allow tofu destroy to delete the S3 bucket even when it contains objects. Set true for dev deployments."
  default     = false
}

variable "lambda_image_tag" {
  type        = string
  description = "ECR image tag to deploy for all Lambda functions (used when lambda_image_digest is not set)"
  default     = "latest"
}

variable "lambda_image_digest" {
  type        = string
  description = "Immutable ECR image digest (e.g. sha256:...) to deploy for all Lambda functions. Takes precedence over lambda_image_tag when set."
  default     = ""
}

variable "tags" {
  type        = map(string)
  description = "Default tags applied to every resource. Change Project here if you want your org name on the resources. The Challenge tag is added automatically from challenge_name."
  default = {
    Project   = "BlindChallenge"
    ManagedBy = "OpenTofu"
  }
}

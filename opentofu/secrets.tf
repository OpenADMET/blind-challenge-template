# -----------------------------------------------------------------------
# Secrets Manager
# -----------------------------------------------------------------------

resource "aws_secretsmanager_secret" "discord_webhook" {
  name        = "${var.challenge_name}-discord-webhook"
  description = "Discord webhook for challenge submission notifications"
}

resource "aws_secretsmanager_secret" "hf_token" {
  name        = "${var.challenge_name}-hf-token"
  description = "HuggingFace API token used by the hf-restarter Lambda"
}

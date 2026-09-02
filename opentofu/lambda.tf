# -----------------------------------------------------------------------
# Regression evaluator
# Shared image: ECR repo ${var.challenge_name}-lambda
# -----------------------------------------------------------------------

resource "aws_lambda_function" "regression_evaluator" {
  function_name                  = "${var.challenge_name}-regression-evaluator"
  role                           = aws_iam_role.lambda_exec.arn
  package_type                   = "Image"
  image_uri                      = var.lambda_image_digest != "" ? "${aws_ecr_repository.lambda.repository_url}@${var.lambda_image_digest}" : "${aws_ecr_repository.lambda.repository_url}:${var.lambda_image_tag}"
  timeout                        = var.lambda_timeout
  memory_size                    = var.lambda_memory_mb
  reserved_concurrent_executions = 5 # cap concurrent evaluation runs

  image_config {
    command = ["backend.lambda_handler_regression.handler"]
  }

  environment {
    variables = {
      S3_BUCKET                   = aws_s3_bucket.main.id
      DISCORD_WEBHOOK_SECRET_NAME = aws_secretsmanager_secret.discord_webhook.name
    }
  }

  depends_on = [aws_cloudwatch_log_group.regression_evaluator]
}

resource "aws_cloudwatch_log_group" "regression_evaluator" {
  name              = "/aws/lambda/${var.challenge_name}-regression-evaluator"
  retention_in_days = 30
}

# -----------------------------------------------------------------------
# Classification evaluator
# Shared image: ECR repo ${var.challenge_name}-lambda
# -----------------------------------------------------------------------

resource "aws_lambda_function" "classification_evaluator" {
  function_name                  = "${var.challenge_name}-classification-evaluator"
  role                           = aws_iam_role.lambda_exec.arn
  package_type                   = "Image"
  image_uri                      = var.lambda_image_digest != "" ? "${aws_ecr_repository.lambda.repository_url}@${var.lambda_image_digest}" : "${aws_ecr_repository.lambda.repository_url}:${var.lambda_image_tag}"
  timeout                        = var.lambda_timeout
  memory_size                    = var.lambda_memory_mb
  reserved_concurrent_executions = 5 # cap concurrent evaluation runs

  image_config {
    command = ["backend.lambda_handler_classification.handler"]
  }

  environment {
    variables = {
      S3_BUCKET                   = aws_s3_bucket.main.id
      DISCORD_WEBHOOK_SECRET_NAME = aws_secretsmanager_secret.discord_webhook.name
    }
  }

  depends_on = [aws_cloudwatch_log_group.classification_evaluator]
}

resource "aws_cloudwatch_log_group" "classification_evaluator" {
  name              = "/aws/lambda/${var.challenge_name}-classification-evaluator"
  retention_in_days = 30
}

# -----------------------------------------------------------------------
# Structure evaluator
# Shared image: ECR repo ${var.challenge_name}-lambda
# -----------------------------------------------------------------------

resource "aws_lambda_function" "structure_evaluator" {
  function_name                  = "${var.challenge_name}-structure-evaluator"
  role                           = aws_iam_role.lambda_exec.arn
  package_type                   = "Image"
  image_uri                      = var.lambda_image_digest != "" ? "${aws_ecr_repository.lambda.repository_url}@${var.lambda_image_digest}" : "${aws_ecr_repository.lambda.repository_url}:${var.lambda_image_tag}"
  timeout                        = var.lambda_timeout
  memory_size                    = var.lambda_memory_mb
  reserved_concurrent_executions = 5 # cap concurrent evaluation runs

  image_config {
    command = ["backend.lambda_handler_structure.handler"]
  }

  environment {
    variables = {
      S3_BUCKET                   = aws_s3_bucket.main.id
      DISCORD_WEBHOOK_SECRET_NAME = aws_secretsmanager_secret.discord_webhook.name
    }
  }

  depends_on = [aws_cloudwatch_log_group.structure_evaluator]
}

resource "aws_cloudwatch_log_group" "structure_evaluator" {
  name              = "/aws/lambda/${var.challenge_name}-structure-evaluator"
  retention_in_days = 30
}

# -----------------------------------------------------------------------
# Leaderboard generator
# Shared image: ECR repo ${var.challenge_name}-lambda
# -----------------------------------------------------------------------

resource "aws_lambda_function" "leaderboard_generator" {
  function_name                  = "${var.challenge_name}-leaderboard-generator"
  role                           = aws_iam_role.lambda_exec.arn
  package_type                   = "Image"
  image_uri                      = var.lambda_image_digest != "" ? "${aws_ecr_repository.lambda.repository_url}@${var.lambda_image_digest}" : "${aws_ecr_repository.lambda.repository_url}:${var.lambda_image_tag}"
  timeout                        = 300
  memory_size                    = 1024
  reserved_concurrent_executions = 1

  image_config {
    command = ["backend.lambda_handler_leaderboard.handler"]
  }

  environment {
    variables = {
      S3_BUCKET                   = aws_s3_bucket.main.id
      DISCORD_WEBHOOK_SECRET_NAME = aws_secretsmanager_secret.discord_webhook.name
    }
  }

  depends_on = [aws_cloudwatch_log_group.leaderboard_generator]
}

resource "aws_cloudwatch_log_group" "leaderboard_generator" {
  name              = "/aws/lambda/${var.challenge_name}-leaderboard-generator"
  retention_in_days = 30
}

# -----------------------------------------------------------------------
# Entries list generator
# Shared image: ECR repo ${var.challenge_name}-lambda
# -----------------------------------------------------------------------

resource "aws_lambda_function" "entries_list_generator" {
  function_name                  = "${var.challenge_name}-entries-list-generator"
  role                           = aws_iam_role.lambda_exec.arn
  package_type                   = "Image"
  image_uri                      = var.lambda_image_digest != "" ? "${aws_ecr_repository.lambda.repository_url}@${var.lambda_image_digest}" : "${aws_ecr_repository.lambda.repository_url}:${var.lambda_image_tag}"
  timeout                        = 300
  memory_size                    = 1024
  reserved_concurrent_executions = 1

  image_config {
    command = ["backend.lambda_handler_entries.handler"]
  }

  environment {
    variables = {
      S3_BUCKET                   = aws_s3_bucket.main.id
      DISCORD_WEBHOOK_SECRET_NAME = aws_secretsmanager_secret.discord_webhook.name
    }
  }

  depends_on = [aws_cloudwatch_log_group.entries_list_generator]
}

resource "aws_cloudwatch_log_group" "entries_list_generator" {
  name              = "/aws/lambda/${var.challenge_name}-entries-list-generator"
  retention_in_days = 30
}

# -----------------------------------------------------------------------
# HuggingFace Space restarter
# Shared image: ECR repo ${var.challenge_name}-lambda
# -----------------------------------------------------------------------

resource "aws_lambda_function" "hf_restarter" {
  function_name = "${var.challenge_name}-hf-restarter"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = var.lambda_image_digest != "" ? "${aws_ecr_repository.lambda.repository_url}@${var.lambda_image_digest}" : "${aws_ecr_repository.lambda.repository_url}:${var.lambda_image_tag}"
  timeout       = 30
  memory_size   = 128

  image_config {
    command = ["backend.lambda_handler_hf_restart.handler"]
  }

  environment {
    variables = {
      HF_OWNER                    = var.hf_owner
      HF_SPACE_NAME               = var.challenge_name
      HF_TOKEN_SECRET_NAME        = aws_secretsmanager_secret.hf_token.name
      DISCORD_WEBHOOK_SECRET_NAME = aws_secretsmanager_secret.discord_webhook.name
    }
  }

  depends_on = [aws_cloudwatch_log_group.hf_restarter]
}

resource "aws_cloudwatch_log_group" "hf_restarter" {
  name              = "/aws/lambda/${var.challenge_name}-hf-restarter"
  retention_in_days = 30
}

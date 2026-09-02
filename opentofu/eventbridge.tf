# -----------------------------------------------------------------------
# Regression evaluator trigger on S3 object create (.csv/.parquet)
# -----------------------------------------------------------------------

resource "aws_cloudwatch_event_rule" "regression_evaluator_s3_object_created" {
  name        = "${var.challenge_name}-regression-object-created"
  description = "Trigger regression evaluator Lambda when regression submissions are uploaded"

  event_pattern = jsonencode({
    source      = ["aws.s3"]
    detail-type = ["Object Created"]
    detail = {
      bucket = {
        name = [aws_s3_bucket.main.id]
      }
      object = {
        key = [
          { wildcard = "submissions/regression/*.csv" },
          { wildcard = "submissions/regression/*.parquet" }
        ]
      }
    }
  })
}

resource "aws_cloudwatch_event_target" "regression_evaluator_s3_object_created" {
  rule      = aws_cloudwatch_event_rule.regression_evaluator_s3_object_created.name
  target_id = "regression-evaluator-lambda-s3-object-created"
  arn       = aws_lambda_function.regression_evaluator.arn
}

resource "aws_lambda_permission" "regression_evaluator_s3_object_created" {
  statement_id  = "AllowEventBridgeS3ObjectCreated"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.regression_evaluator.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.regression_evaluator_s3_object_created.arn
}

# -----------------------------------------------------------------------
# Classification evaluator trigger on S3 object create (.csv/.parquet)
# -----------------------------------------------------------------------

resource "aws_cloudwatch_event_rule" "classification_evaluator_s3_object_created" {
  name        = "${var.challenge_name}-classification-object-created"
  description = "Trigger classification evaluator Lambda when classification submissions are uploaded"

  event_pattern = jsonencode({
    source      = ["aws.s3"]
    detail-type = ["Object Created"]
    detail = {
      bucket = {
        name = [aws_s3_bucket.main.id]
      }
      object = {
        key = [
          { wildcard = "submissions/classification/*.csv" },
          { wildcard = "submissions/classification/*.parquet" }
        ]
      }
    }
  })
}

resource "aws_cloudwatch_event_target" "classification_evaluator_s3_object_created" {
  rule      = aws_cloudwatch_event_rule.classification_evaluator_s3_object_created.name
  target_id = "classification-evaluator-lambda-s3-object-created"
  arn       = aws_lambda_function.classification_evaluator.arn
}

resource "aws_lambda_permission" "classification_evaluator_s3_object_created" {
  statement_id  = "AllowEventBridgeClassificationS3ObjectCreated"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.classification_evaluator.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.classification_evaluator_s3_object_created.arn
}

# -----------------------------------------------------------------------
# Structure evaluator trigger on S3 object create (.zip)
# -----------------------------------------------------------------------

resource "aws_cloudwatch_event_rule" "structure_evaluator_s3_object_created" {
  name        = "${var.challenge_name}-structure-object-created"
  description = "Trigger structure evaluator Lambda when structure submissions are uploaded"

  event_pattern = jsonencode({
    source      = ["aws.s3"]
    detail-type = ["Object Created"]
    detail = {
      bucket = {
        name = [aws_s3_bucket.main.id]
      }
      object = {
        key = [
          { wildcard = "submissions/structure/*.zip" }
        ]
      }
    }
  })
}

resource "aws_cloudwatch_event_target" "structure_evaluator_s3_object_created" {
  rule      = aws_cloudwatch_event_rule.structure_evaluator_s3_object_created.name
  target_id = "structure-evaluator-lambda-s3-object-created"
  arn       = aws_lambda_function.structure_evaluator.arn
}

resource "aws_lambda_permission" "structure_evaluator_s3_object_created" {
  statement_id  = "AllowEventBridgeStructureS3ObjectCreated"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.structure_evaluator.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.structure_evaluator_s3_object_created.arn
}

# -----------------------------------------------------------------------
# Leaderboard generator schedule
# -----------------------------------------------------------------------

resource "aws_cloudwatch_event_rule" "leaderboard_generator" {
  name                = "${var.challenge_name}-leaderboard-generator"
  description         = "Trigger leaderboard generator Lambda"
  schedule_expression = var.leaderboard_schedule
}

resource "aws_cloudwatch_event_target" "leaderboard_generator" {
  rule      = aws_cloudwatch_event_rule.leaderboard_generator.name
  target_id = "leaderboard-generator-lambda"
  arn       = aws_lambda_function.leaderboard_generator.arn
}

resource "aws_lambda_permission" "leaderboard_generator" {
  statement_id  = "AllowEventBridgeLeaderboard"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.leaderboard_generator.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.leaderboard_generator.arn
}

# -----------------------------------------------------------------------
# Entries list generator schedule
# -----------------------------------------------------------------------

resource "aws_cloudwatch_event_rule" "entries_list_generator" {
  name                = "${var.challenge_name}-entries-list-generator"
  description         = "Trigger entries list generator Lambda (daily, ~6am NZT)"
  schedule_expression = var.entries_list_schedule
}

resource "aws_cloudwatch_event_target" "entries_list_generator" {
  rule      = aws_cloudwatch_event_rule.entries_list_generator.name
  target_id = "entries-list-generator-lambda"
  arn       = aws_lambda_function.entries_list_generator.arn
}

resource "aws_lambda_permission" "entries_list_generator" {
  statement_id  = "AllowEventBridgeEntriesList"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.entries_list_generator.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.entries_list_generator.arn
}

# -----------------------------------------------------------------------
# HuggingFace Space restarter schedule
# -----------------------------------------------------------------------

resource "aws_cloudwatch_event_rule" "hf_restarter" {
  name                = "${var.challenge_name}-hf-restarter"
  description         = "Periodically restart the HuggingFace Space"
  schedule_expression = var.hf_restart_schedule
}

resource "aws_cloudwatch_event_target" "hf_restarter" {
  rule      = aws_cloudwatch_event_rule.hf_restarter.name
  target_id = "hf-restarter-lambda"
  arn       = aws_lambda_function.hf_restarter.arn
}

resource "aws_lambda_permission" "hf_restarter" {
  statement_id  = "AllowEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.hf_restarter.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.hf_restarter.arn
}

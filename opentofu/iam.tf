# -----------------------------------------------------------------------
# IAM user for the HuggingFace Space (submit to S3, read leaderboards)
# -----------------------------------------------------------------------

resource "aws_iam_user" "hf_space" {
  name = "${var.challenge_name}-hf-space"
}

resource "aws_iam_access_key" "hf_space" {
  user = aws_iam_user.hf_space.name
}

resource "aws_iam_user_policy" "hf_space" {
  name = "${var.challenge_name}-hf-space-policy"
  user = aws_iam_user.hf_space.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "PutSubmissions"
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = "${aws_s3_bucket.main.arn}/submissions/*"
      },
      {
        Sid    = "GetChallengeData"
        Effect = "Allow"
        Action = ["s3:GetObject"]
        Resource = [
          "${aws_s3_bucket.main.arn}/leaderboard/*",
          "${aws_s3_bucket.main.arn}/submissions/*"
        ]
      },
      {
        Sid      = "ListSubmissionPrefixes"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.main.arn
        Condition = {
          StringLike = {
            "s3:prefix" = [
              "submissions/regression/*",
              "submissions/classification/*",
              "submissions/structure/*",
              "leaderboard",
              "leaderboard/",
              "leaderboard/*",
              "leaderboard/live/*",
              "leaderboard/interim/*",
              "leaderboard/final/*"
            ]
          }
        }
      }
    ]
  })
}

# -----------------------------------------------------------------------
# IAM user for CI/CD or deploy workflows (push Lambda images to ECR)
# -----------------------------------------------------------------------

resource "aws_iam_user" "artifacts_uploader" {
  name = "${var.challenge_name}-artifacts-uploader"
}

resource "aws_iam_access_key" "artifacts_uploader" {
  user = aws_iam_user.artifacts_uploader.name
}

resource "aws_iam_user_policy" "artifacts_uploader" {
  name = "${var.challenge_name}-artifacts-uploader-policy"
  user = aws_iam_user.artifacts_uploader.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "EcrAuth"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Sid    = "PushEvaluatorImages"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:CompleteLayerUpload",
          "ecr:InitiateLayerUpload",
          "ecr:PutImage",
          "ecr:UploadLayerPart",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer"
        ]
        Resource = [
          aws_ecr_repository.lambda.arn,
        ]
      }
    ]
  })
}

# -----------------------------------------------------------------------
# Shared Lambda execution role (used by all Lambda functions)
# -----------------------------------------------------------------------

resource "aws_iam_role" "lambda_exec" {
  name = "${var.challenge_name}-lambda-exec"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_s3" {
  name = "${var.challenge_name}-lambda-s3-policy"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadSubmissionsAndGroundTruth"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.main.arn,
          "${aws_s3_bucket.main.arn}/submissions/*",
          "${aws_s3_bucket.main.arn}/ground_truth/*",
          "${aws_s3_bucket.main.arn}/scores/*",
          "${aws_s3_bucket.main.arn}/leaderboard/*",
          "${aws_s3_bucket.main.arn}/entries/*",
        ]
      },
      {
        Sid    = "WriteScoresAndLeaderboards"
        Effect = "Allow"
        Action = ["s3:PutObject"]
        Resource = [
          "${aws_s3_bucket.main.arn}/submissions/*/scores.json",
          "${aws_s3_bucket.main.arn}/submissions/manifest/*",
          "${aws_s3_bucket.main.arn}/scores/*",
          "${aws_s3_bucket.main.arn}/leaderboard/*",
          "${aws_s3_bucket.main.arn}/entries/*",
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy" "lambda_secrets" {
  name = "${var.challenge_name}-lambda-secrets-policy"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadChallengeSecrets"
        Effect = "Allow"
        Action = [
          "secretsmanager:DescribeSecret",
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.discord_webhook.arn,
          aws_secretsmanager_secret.hf_token.arn,
        ]
      }
    ]
  })
}

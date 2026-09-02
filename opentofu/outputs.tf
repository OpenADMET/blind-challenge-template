output "hf_space_access_key_id" {
  description = "AWS access key ID for the HuggingFace Space IAM user. Set as the AWS_ACCESS_KEY_ID Space secret."
  value       = aws_iam_access_key.hf_space.id
  sensitive   = true
}

output "hf_space_secret_access_key" {
  description = "AWS secret access key for the HuggingFace Space IAM user. Set as the AWS_SECRET_ACCESS_KEY Space secret."
  value       = aws_iam_access_key.hf_space.secret
  sensitive   = true
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket. Set as the S3_BUCKET Space secret."
  value       = aws_s3_bucket.main.id
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket."
  value       = aws_s3_bucket.main.arn
}

output "artifacts_uploader_access_key_id" {
  description = "AWS access key ID for the artifacts uploader IAM user. Use for pushing Lambda images to ECR."
  value       = aws_iam_access_key.artifacts_uploader.id
  sensitive   = true
}

output "artifacts_uploader_secret_access_key" {
  description = "AWS secret access key for the artifacts uploader IAM user."
  value       = aws_iam_access_key.artifacts_uploader.secret
  sensitive   = true
}

output "lambda_ecr_repository_url" {
  description = "ECR repository URL for all image-based Lambda functions."
  value       = aws_ecr_repository.lambda.repository_url
}

output "discord_webhook_secret_name" {
  description = "Secrets Manager secret name for the Discord webhook URL."
  value       = aws_secretsmanager_secret.discord_webhook.name
}

output "discord_webhook_secret_arn" {
  description = "Secrets Manager secret ARN for the Discord webhook URL."
  value       = aws_secretsmanager_secret.discord_webhook.arn
}

output "hf_token_secret_name" {
  description = "Secrets Manager secret name for the HuggingFace API token."
  value       = aws_secretsmanager_secret.hf_token.name
}

output "hf_token_secret_arn" {
  description = "Secrets Manager secret ARN for the HuggingFace API token."
  value       = aws_secretsmanager_secret.hf_token.arn
}

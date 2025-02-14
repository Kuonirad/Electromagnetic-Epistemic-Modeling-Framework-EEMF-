terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-west-2"
}

resource "aws_lambda_function" "worker" {
  filename         = "lambda.zip"
  function_name    = "em-sim-worker"
  role            = aws_iam_role.lambda_role.arn
  handler         = "worker.handler"
  runtime         = "python3.12"
  timeout         = 900

  environment {
    variables = {
      STAGE = "prod"
    }
  }

  ephemeral_storage {
    size = 10240 # MB
  }
}

resource "aws_iam_role" "lambda_role" {
  name = "em_sim_lambda_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

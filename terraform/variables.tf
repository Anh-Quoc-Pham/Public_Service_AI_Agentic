variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "public-service-navigation"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24"]
}

variable "task_cpu" {
  description = "CPU units for the ECS task"
  type        = number
  default     = 1024
}

variable "task_memory" {
  description = "Memory for the ECS task (MiB)"
  type        = number
  default     = 2048
}

variable "service_desired_count" {
  description = "Desired number of ECS service instances"
  type        = number
  default     = 1
}

variable "llm_model" {
  description = "LLM model to use"
  type        = string
  default     = "mixtral-8x7b"
}

variable "common_tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project     = "public-service-navigation"
    Environment = "production"
    ManagedBy   = "terraform"
  }
} 
terraform {
  required_providers {
    databricks = {
      source = "databricks/databricks"
    }
  }
}

# Auto-detect the identity running `databricks bundle deploy`.
# This is a service principal during CI/CD and a user during local deploys.
data "databricks_current_user" "deployer" {}

variable "secret_acl_principal" {
  description = "Override the principal that receives secret READ access. Defaults to the deploying identity when left empty."
  type        = string
  default     = ""
}

locals {
  secret_scopes = [
    "databricks-evaluation-app",
    "agent-obo-scope",
  ]
  # Use the explicit override when provided; otherwise fall back to the
  # identity that is running `databricks bundle deploy`.
  principal = trimspace(var.secret_acl_principal) != "" ? trimspace(var.secret_acl_principal) : data.databricks_current_user.deployer.user_name
}

resource "databricks_secret_acl" "app_secret_read" {
  for_each = toset(local.secret_scopes)

  scope      = each.value
  principal  = local.principal
  permission = "READ"
}

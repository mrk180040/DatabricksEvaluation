terraform {
  required_providers {
    databricks = {
      source = "databricks/databricks"
    }
  }
}

variable "secret_acl_principal" {
  description = "Principal to grant secret READ access (user/group/service principal application id)."
  type        = string
  default     = ""
}

locals {
  secret_scopes = [
    "databricks-evaluation-app",
    "agent-obo-scope",
  ]
  apply_secret_acls = trim(var.secret_acl_principal) != ""
}

resource "databricks_secret_acl" "app_secret_read" {
  for_each = local.apply_secret_acls ? toset(local.secret_scopes) : toset([])

  scope      = each.value
  principal  = var.secret_acl_principal
  permission = "READ"
}

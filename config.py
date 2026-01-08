import os
from dotenv import load_dotenv

load_dotenv()

# Path to your Google Cloud service account JSON file (optional)
# Leave empty or unset to use user credentials via gcloud
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')

# Default GCP Project ID (used for the BigQuery client)
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', None)

# List of GCP projects to compare tables across
# Comma-separated in .env, e.g.: GCP_PROJECTS=tecovas-staging,tecovas-development
_projects_str = os.getenv('GCP_PROJECTS', '')
GCP_PROJECTS = [p.strip() for p in _projects_str.split(',') if p.strip()]

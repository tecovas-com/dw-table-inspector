import os
from google.cloud import bigquery
import config

# Cache the client instance
_client = None


def get_client():
    """
    Initialize and return a BigQuery client.

    Uses credentials in this order:
    1. Service account file if GOOGLE_APPLICATION_CREDENTIALS is set
    2. Application Default Credentials (gcloud auth login)
    """
    global _client
    if _client is not None:
        return _client

    creds_path = config.GOOGLE_APPLICATION_CREDENTIALS

    # If a service account file is specified and exists, use it
    if creds_path and os.path.exists(creds_path):
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(creds_path)
        project_id = config.GCP_PROJECT_ID or credentials.project_id
        _client = bigquery.Client(credentials=credentials, project=project_id)
    else:
        # Use Application Default Credentials (user credentials via gcloud)
        _client = bigquery.Client(project=config.GCP_PROJECT_ID)

    return _client


def list_projects():
    """List configured projects."""
    return config.GCP_PROJECTS


def list_datasets(project_id):
    """List all datasets in a project."""
    client = get_client()
    datasets = list(client.list_datasets(project=project_id))
    return [dataset.dataset_id for dataset in datasets]


def list_tables(project_id, dataset_id):
    """List all tables in a dataset."""
    client = get_client()
    dataset_ref = f"{project_id}.{dataset_id}"
    tables = list(client.list_tables(dataset_ref))
    return [table.table_id for table in tables]


def get_table_schema(project_id, dataset_id, table_id):
    """Get the schema of a table."""
    client = get_client()
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    table = client.get_table(table_ref)

    schema = []
    for field in table.schema:
        schema.append({
            'name': field.name,
            'type': field.field_type,
            'mode': field.mode,
            'description': field.description or ''
        })

    return schema


def get_row_count(project_id, dataset_id, table_id):
    """Get the row count of a table."""
    client = get_client()
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    table = client.get_table(table_ref)
    return table.num_rows


def get_column_stats(project_id, dataset_id, table_id, columns):
    """
    Get statistics for specified columns.
    Returns min, max, null count, and distinct count for each column.
    """
    client = get_client()
    table_ref = f"`{project_id}.{dataset_id}.{table_id}`"

    stats = {}

    for col in columns:
        col_name = col['name']
        col_type = col['type']

        # Build query based on column type
        if col_type in ('STRING', 'INT64', 'FLOAT64', 'NUMERIC', 'BIGNUMERIC', 'DATE', 'DATETIME', 'TIMESTAMP'):
            query = f"""
                SELECT
                    MIN(`{col_name}`) as min_val,
                    MAX(`{col_name}`) as max_val,
                    COUNTIF(`{col_name}` IS NULL) as null_count,
                    COUNT(DISTINCT `{col_name}`) as distinct_count
                FROM {table_ref}
            """
        else:
            # For complex types, just get null and non-null counts
            query = f"""
                SELECT
                    NULL as min_val,
                    NULL as max_val,
                    COUNTIF(`{col_name}` IS NULL) as null_count,
                    NULL as distinct_count
                FROM {table_ref}
            """

        try:
            result = client.query(query).result()
            row = list(result)[0]
            stats[col_name] = {
                'min': str(row.min_val) if row.min_val is not None else None,
                'max': str(row.max_val) if row.max_val is not None else None,
                'null_count': row.null_count,
                'distinct_count': row.distinct_count
            }
        except Exception as e:
            stats[col_name] = {
                'min': None,
                'max': None,
                'null_count': None,
                'distinct_count': None,
                'error': str(e)
            }

    return stats


def find_row_differences(project1, dataset1, table1, project2, dataset2, table2, primary_key, limit=100):
    """
    Find rows that exist in one table but not the other, or have different values.
    Returns sample of differing rows.
    """
    client = get_client()
    table1_ref = f"`{project1}.{dataset1}.{table1}`"
    table2_ref = f"`{project2}.{dataset2}.{table2}`"

    results = {
        'only_in_table1': [],
        'only_in_table2': [],
        'different_values': []
    }

    # Find rows only in table 1
    query_only_in_1 = f"""
        SELECT t1.*
        FROM {table1_ref} t1
        LEFT JOIN {table2_ref} t2 ON t1.`{primary_key}` = t2.`{primary_key}`
        WHERE t2.`{primary_key}` IS NULL
        LIMIT {limit}
    """

    try:
        rows = client.query(query_only_in_1).result()
        for row in rows:
            results['only_in_table1'].append(dict(row))
    except Exception as e:
        results['only_in_table1_error'] = str(e)

    # Find rows only in table 2
    query_only_in_2 = f"""
        SELECT t2.*
        FROM {table2_ref} t2
        LEFT JOIN {table1_ref} t1 ON t2.`{primary_key}` = t1.`{primary_key}`
        WHERE t1.`{primary_key}` IS NULL
        LIMIT {limit}
    """

    try:
        rows = client.query(query_only_in_2).result()
        for row in rows:
            results['only_in_table2'].append(dict(row))
    except Exception as e:
        results['only_in_table2_error'] = str(e)

    # Find rows with different values (using EXCEPT)
    # This finds rows where the primary key exists in both but values differ
    query_diff = f"""
        WITH matched_keys AS (
            SELECT t1.`{primary_key}` as pk
            FROM {table1_ref} t1
            INNER JOIN {table2_ref} t2 ON t1.`{primary_key}` = t2.`{primary_key}`
        ),
        t1_rows AS (
            SELECT * FROM {table1_ref} WHERE `{primary_key}` IN (SELECT pk FROM matched_keys)
        ),
        t2_rows AS (
            SELECT * FROM {table2_ref} WHERE `{primary_key}` IN (SELECT pk FROM matched_keys)
        ),
        diff_keys AS (
            SELECT `{primary_key}` FROM t1_rows
            EXCEPT DISTINCT
            SELECT `{primary_key}` FROM t2_rows
        )
        SELECT t1.`{primary_key}`, 'table1' as source, t1.* EXCEPT(`{primary_key}`)
        FROM {table1_ref} t1
        WHERE t1.`{primary_key}` IN (SELECT `{primary_key}` FROM diff_keys)
        UNION ALL
        SELECT t2.`{primary_key}`, 'table2' as source, t2.* EXCEPT(`{primary_key}`)
        FROM {table2_ref} t2
        WHERE t2.`{primary_key}` IN (SELECT `{primary_key}` FROM diff_keys)
        ORDER BY `{primary_key}`, source
        LIMIT {limit * 2}
    """

    try:
        rows = client.query(query_diff).result()
        for row in rows:
            results['different_values'].append(dict(row))
    except Exception as e:
        results['different_values_error'] = str(e)

    return results

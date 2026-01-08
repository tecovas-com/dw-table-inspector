import os
import time
import json
import logging
from datetime import datetime
from functools import wraps
from google.cloud import bigquery
import config

# Cache the client instance
_client = None

# =============================================================================
# LOGGING & INSTRUMENTATION
# =============================================================================

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('bigquery_client')

# Log file path (JSONL format - one JSON object per line)
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'logs', 'bq_requests.jsonl')

# Enable/disable detailed console logging (set via environment variable)
DEBUG_MODE = os.getenv('BQ_DEBUG', 'false').lower() == 'true'


def _ensure_log_dir():
    """Ensure the log directory exists."""
    log_dir = os.path.dirname(LOG_FILE_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)


def _append_log_entry(entry):
    """Append a single log entry as a JSON line."""
    _ensure_log_dir()
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(json.dumps(entry, default=str) + '\n')


def log_request(func_name, params, query=None):
    """Log the start of a request."""
    entry = {
        'timestamp': datetime.now().astimezone().isoformat(),
        'function': func_name,
        'params': params,
        'query': query,
        'status': 'started'
    }
    if DEBUG_MODE:
        logger.info(f"[BQ REQUEST] {func_name} - params: {params}")
        if query:
            logger.info(f"[BQ QUERY]\n{query}")
    return entry


def log_response(entry, duration_ms, rows=None, bytes_processed=None, error=None):
    """Log the completion of a request by appending to file."""
    entry['duration_ms'] = round(duration_ms, 2)
    entry['rows_returned'] = rows
    entry['bytes_processed'] = bytes_processed
    entry['bytes_processed_mb'] = round(bytes_processed / 1_000_000, 2) if bytes_processed else None
    entry['status'] = 'error' if error else 'success'
    entry['error'] = str(error) if error else None

    # Append single line to log file
    _append_log_entry(entry)

    if DEBUG_MODE:
        if error:
            logger.error(f"[BQ ERROR] {entry['function']} - {error} ({duration_ms:.0f}ms)")
        else:
            logger.info(f"[BQ RESPONSE] {entry['function']} - {rows} rows, {bytes_processed or 'N/A'} bytes ({duration_ms:.0f}ms)")

    return entry


def get_request_log():
    """Get all logged requests from file."""
    if not os.path.exists(LOG_FILE_PATH):
        return []
    requests = []
    with open(LOG_FILE_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                requests.append(json.loads(line))
    return requests


def clear_request_log():
    """Clear the request log file."""
    _ensure_log_dir()
    with open(LOG_FILE_PATH, 'w') as f:
        pass  # Truncate file


def get_request_summary():
    """Get a summary of all requests."""
    requests = get_request_log()
    if not requests:
        return {'total_requests': 0}

    total_time = sum(r.get('duration_ms', 0) for r in requests)
    total_bytes = sum(r.get('bytes_processed', 0) or 0 for r in requests)
    errors = [r for r in requests if r.get('status') == 'error']

    by_function = {}
    for r in requests:
        func = r['function']
        if func not in by_function:
            by_function[func] = {'count': 0, 'total_ms': 0}
        by_function[func]['count'] += 1
        by_function[func]['total_ms'] += r.get('duration_ms', 0)

    for func in by_function:
        by_function[func]['avg_ms'] = round(
            by_function[func]['total_ms'] / by_function[func]['count'], 2
        )
        by_function[func]['total_ms'] = round(by_function[func]['total_ms'], 2)

    return {
        'total_requests': len(requests),
        'total_time_ms': round(total_time, 2),
        'total_bytes_processed': total_bytes,
        'total_bytes_processed_mb': round(total_bytes / 1_000_000, 2),
        'errors': len(errors),
        'by_function': by_function
    }


# =============================================================================
# BIGQUERY CLIENT
# =============================================================================


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
    entry = log_request('list_datasets', {'project_id': project_id})
    start = time.time()

    try:
        client = get_client()
        datasets = list(client.list_datasets(project=project_id))
        result = [dataset.dataset_id for dataset in datasets]
        log_response(entry, (time.time() - start) * 1000, rows=len(result))
        return result
    except Exception as e:
        log_response(entry, (time.time() - start) * 1000, error=e)
        raise


def list_tables(project_id, dataset_id):
    """List all tables in a dataset."""
    entry = log_request('list_tables', {'project_id': project_id, 'dataset_id': dataset_id})
    start = time.time()

    try:
        client = get_client()
        dataset_ref = f"{project_id}.{dataset_id}"
        tables = list(client.list_tables(dataset_ref))
        result = [table.table_id for table in tables]
        log_response(entry, (time.time() - start) * 1000, rows=len(result))
        return result
    except Exception as e:
        log_response(entry, (time.time() - start) * 1000, error=e)
        raise


def get_table_schema(project_id, dataset_id, table_id):
    """Get the schema of a table."""
    entry = log_request('get_table_schema', {
        'project_id': project_id, 'dataset_id': dataset_id, 'table_id': table_id
    })
    start = time.time()

    try:
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

        log_response(entry, (time.time() - start) * 1000, rows=len(schema))
        return schema
    except Exception as e:
        log_response(entry, (time.time() - start) * 1000, error=e)
        raise


def get_row_count(project_id, dataset_id, table_id):
    """Get the row count of a table."""
    entry = log_request('get_row_count', {
        'project_id': project_id, 'dataset_id': dataset_id, 'table_id': table_id
    })
    start = time.time()

    try:
        client = get_client()
        table_ref = f"{project_id}.{dataset_id}.{table_id}"
        table = client.get_table(table_ref)
        log_response(entry, (time.time() - start) * 1000, rows=1)
        return table.num_rows
    except Exception as e:
        log_response(entry, (time.time() - start) * 1000, error=e)
        raise


def get_column_stats(project_id, dataset_id, table_id, columns):
    """
    Get statistics for specified columns in a single query.
    Returns min, max, null count, and distinct count for each column.
    """
    client = get_client()
    table_ref = f"`{project_id}.{dataset_id}.{table_id}`"

    # Supported types for min/max/distinct
    supported_types = ('STRING', 'INT64', 'FLOAT64', 'NUMERIC', 'BIGNUMERIC', 'DATE', 'DATETIME', 'TIMESTAMP')

    # Build SELECT clause for all columns
    select_parts = []
    for col in columns:
        col_name = col['name']
        col_type = col['type']

        if col_type in supported_types:
            select_parts.append(f"MIN(`{col_name}`) as `{col_name}_min`")
            select_parts.append(f"MAX(`{col_name}`) as `{col_name}_max`")
            select_parts.append(f"COUNTIF(`{col_name}` IS NULL) as `{col_name}_nulls`")
            select_parts.append(f"COUNT(DISTINCT `{col_name}`) as `{col_name}_distinct`")
        else:
            # For complex types, just get null count
            select_parts.append(f"NULL as `{col_name}_min`")
            select_parts.append(f"NULL as `{col_name}_max`")
            select_parts.append(f"COUNTIF(`{col_name}` IS NULL) as `{col_name}_nulls`")
            select_parts.append(f"NULL as `{col_name}_distinct`")

    query = f"SELECT\n    {',\n    '.join(select_parts)}\nFROM {table_ref}"

    entry = log_request('get_column_stats', {
        'table': f"{project_id}.{dataset_id}.{table_id}",
        'columns': [c['name'] for c in columns]
    }, query=query)
    start = time.time()

    stats = {}
    try:
        job = client.query(query)
        result = job.result()
        row = list(result)[0]

        for col in columns:
            col_name = col['name']
            min_val = getattr(row, f"{col_name}_min", None)
            max_val = getattr(row, f"{col_name}_max", None)
            stats[col_name] = {
                'min': str(min_val) if min_val is not None else None,
                'max': str(max_val) if max_val is not None else None,
                'null_count': getattr(row, f"{col_name}_nulls", None),
                'distinct_count': getattr(row, f"{col_name}_distinct", None)
            }

        log_response(entry, (time.time() - start) * 1000,
                    rows=1, bytes_processed=job.total_bytes_processed)
    except Exception as e:
        for col in columns:
            stats[col['name']] = {
                'min': None,
                'max': None,
                'null_count': None,
                'distinct_count': None,
                'error': str(e)
            }
        log_response(entry, (time.time() - start) * 1000, error=e)

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

    entry = log_request('find_row_differences:only_in_table1', {
        'table1': f"{project1}.{dataset1}.{table1}",
        'table2': f"{project2}.{dataset2}.{table2}",
        'primary_key': primary_key
    }, query=query_only_in_1.strip())
    start = time.time()

    try:
        job = client.query(query_only_in_1)
        rows = job.result()
        for row in rows:
            results['only_in_table1'].append(dict(row))
        log_response(entry, (time.time() - start) * 1000,
                    rows=len(results['only_in_table1']),
                    bytes_processed=job.total_bytes_processed)
    except Exception as e:
        results['only_in_table1_error'] = str(e)
        log_response(entry, (time.time() - start) * 1000, error=e)

    # Find rows only in table 2
    query_only_in_2 = f"""
        SELECT t2.*
        FROM {table2_ref} t2
        LEFT JOIN {table1_ref} t1 ON t2.`{primary_key}` = t1.`{primary_key}`
        WHERE t1.`{primary_key}` IS NULL
        LIMIT {limit}
    """

    entry = log_request('find_row_differences:only_in_table2', {
        'table1': f"{project1}.{dataset1}.{table1}",
        'table2': f"{project2}.{dataset2}.{table2}",
        'primary_key': primary_key
    }, query=query_only_in_2.strip())
    start = time.time()

    try:
        job = client.query(query_only_in_2)
        rows = job.result()
        for row in rows:
            results['only_in_table2'].append(dict(row))
        log_response(entry, (time.time() - start) * 1000,
                    rows=len(results['only_in_table2']),
                    bytes_processed=job.total_bytes_processed)
    except Exception as e:
        results['only_in_table2_error'] = str(e)
        log_response(entry, (time.time() - start) * 1000, error=e)

    # Find rows with different values (using EXCEPT)
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

    entry = log_request('find_row_differences:diff_values', {
        'table1': f"{project1}.{dataset1}.{table1}",
        'table2': f"{project2}.{dataset2}.{table2}",
        'primary_key': primary_key
    }, query=query_diff.strip())
    start = time.time()

    try:
        job = client.query(query_diff)
        rows = job.result()
        for row in rows:
            results['different_values'].append(dict(row))
        log_response(entry, (time.time() - start) * 1000,
                    rows=len(results['different_values']),
                    bytes_processed=job.total_bytes_processed)
    except Exception as e:
        results['different_values_error'] = str(e)
        log_response(entry, (time.time() - start) * 1000, error=e)

    return results

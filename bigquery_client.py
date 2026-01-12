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

# Columns to exclude from comparisons (Fivetran metadata columns)
EXCLUDED_COLUMNS = {'_fivetran_id', '_fivetran_synced', '_dbt_loaded_at'}

# Column types to exclude from comparisons (not supported in EXCEPT DISTINCT or complex to compare)
EXCLUDED_TYPES = {'STRUCT', 'RECORD', 'ARRAY'}

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


def get_row_count(project_id, dataset_id, table_id, where_filter=None):
    """Get the row count of a table by running an actual COUNT query."""
    query = f"SELECT COUNT(*) as cnt FROM `{project_id}.{dataset_id}.{table_id}`"
    if where_filter:
        query += f" WHERE {where_filter}"

    entry = log_request('get_row_count', {
        'project_id': project_id, 'dataset_id': dataset_id, 'table_id': table_id,
        'where_filter': where_filter
    }, query=query)
    start = time.time()

    try:
        client = get_client()
        job = client.query(query)
        result = list(job.result())[0]
        log_response(entry, (time.time() - start) * 1000,
                    rows=1, bytes_processed=job.total_bytes_processed)
        return result.cnt
    except Exception as e:
        log_response(entry, (time.time() - start) * 1000, error=e)
        raise


def get_row_count_query(project_id, dataset_id, table_id, where_filter=None):
    """Return the SQL query for getting row count."""
    query = f"SELECT COUNT(*) as row_count FROM `{project_id}.{dataset_id}.{table_id}`"
    if where_filter:
        query += f" WHERE {where_filter}"
    return query


def get_column_stats(project_id, dataset_id, table_id, columns, where_filter=None):
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
    if where_filter:
        query += f"\nWHERE {where_filter}"

    entry = log_request('get_column_stats', {
        'table': f"{project_id}.{dataset_id}.{table_id}",
        'columns': [c['name'] for c in columns],
        'where_filter': where_filter
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


def find_row_differences(project1, dataset1, table1, project2, dataset2, table2, primary_key, limit=100, where_filter=None):
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
        'different_values': [],
        'queries': {}
    }

    # Handle multiple primary keys
    pk_columns = [col.strip() for col in primary_key.split(',')]

    # Get schemas to find common columns for the diff query
    # Exclude: Fivetran columns and STRUCT/ARRAY types (not supported in comparisons)
    schema1 = get_table_schema(project1, dataset1, table1)
    schema2 = get_table_schema(project2, dataset2, table2)

    # Build dicts with name -> type mapping, excluding Fivetran columns and unsupported types
    schema1_dict = {
        col['name']: col['type'] for col in schema1
        if col['name'] not in EXCLUDED_COLUMNS and col['type'] not in EXCLUDED_TYPES
    }
    schema2_dict = {
        col['name']: col['type'] for col in schema2
        if col['name'] not in EXCLUDED_COLUMNS and col['type'] not in EXCLUDED_TYPES
    }
    common_columns = sorted(set(schema1_dict.keys()) & set(schema2_dict.keys()))

    # Build list of non-PK common columns for SELECT
    non_pk_common_columns = [col for col in common_columns if col not in pk_columns]

    # Build column select list for CTEs (common columns only)
    common_columns_sql = ', '.join([f'`{col}`' for col in common_columns])

    # Build JOIN condition for multiple primary keys
    join_conditions = ' AND '.join([f"t1.`{col}` = t2.`{col}`" for col in pk_columns])
    
    # Function to generate WHERE condition for NULL check (any key column being NULL means row doesn't exist)
    def where_null_condition(alias):
        return ' OR '.join([f"{alias}.`{col}` IS NULL" for col in pk_columns])

    # Find rows only in table 1
    # Use CTEs to apply filter before JOIN to avoid ambiguous column references
    if where_filter:
        query_only_in_1 = f"""
            WITH filtered_t1 AS (
                SELECT * FROM {table1_ref} WHERE {where_filter}
            ),
            filtered_t2 AS (
                SELECT * FROM {table2_ref} WHERE {where_filter}
            )
            SELECT t1.*
            FROM filtered_t1 t1
            LEFT JOIN filtered_t2 t2 ON {join_conditions}
            WHERE {where_null_condition('t2')}
            LIMIT {limit}
        """
    else:
        query_only_in_1 = f"""
            SELECT t1.*
            FROM {table1_ref} t1
            LEFT JOIN {table2_ref} t2 ON {join_conditions}
            WHERE {where_null_condition('t2')}
            LIMIT {limit}
        """

    entry = log_request('find_row_differences:only_in_table1', {
        'table1': f"{project1}.{dataset1}.{table1}",
        'table2': f"{project2}.{dataset2}.{table2}",
        'primary_key': primary_key,
        'where_filter': where_filter
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
    if where_filter:
        query_only_in_2 = f"""
            WITH filtered_t1 AS (
                SELECT * FROM {table1_ref} WHERE {where_filter}
            ),
            filtered_t2 AS (
                SELECT * FROM {table2_ref} WHERE {where_filter}
            )
            SELECT t2.*
            FROM filtered_t2 t2
            LEFT JOIN filtered_t1 t1 ON {join_conditions}
            WHERE {where_null_condition('t1')}
            LIMIT {limit}
        """
    else:
        query_only_in_2 = f"""
            SELECT t2.*
            FROM {table2_ref} t2
            LEFT JOIN {table1_ref} t1 ON {join_conditions}
            WHERE {where_null_condition('t1')}
            LIMIT {limit}
        """

    entry = log_request('find_row_differences:only_in_table2', {
        'table1': f"{project1}.{dataset1}.{table1}",
        'table2': f"{project2}.{dataset2}.{table2}",
        'primary_key': primary_key,
        'where_filter': where_filter
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

    # Find rows with different values (using JOIN with column comparisons)
    # Build SELECT list for primary key columns
    def pk_select_list(alias):
        return ', '.join([f"{alias}.`{col}`" for col in pk_columns])

    # Build SELECT list for non-PK common columns with source/target prefixes
    def non_pk_select_list_both():
        if non_pk_common_columns:
            parts = []
            for col in non_pk_common_columns:
                parts.append(f"t1.`{col}` as `source_{col}`")
                parts.append(f"t2.`{col}` as `target_{col}`")
            return ', ' + ', '.join(parts)
        return ''

    # Build ORDER BY for primary keys
    order_by_list = ', '.join([f"`{col}`" for col in pk_columns])

    # Build WHERE condition for column differences (NULL = NULL is considered equal)
    # Condition: values are different if NOT (t1.col = t2.col OR (t1.col IS NULL AND t2.col IS NULL))
    def column_diff_conditions():
        if non_pk_common_columns:
            conditions = []
            for col in non_pk_common_columns:
                conditions.append(f"NOT (t1.`{col}` = t2.`{col}` OR (t1.`{col}` IS NULL AND t2.`{col}` IS NULL))")
            return ' OR '.join(conditions)
        return '1=0'  # No columns to compare

    if where_filter:
        query_diff = f"""
            WITH filtered_t1 AS (
                SELECT * FROM {table1_ref} WHERE {where_filter}
            ),
            filtered_t2 AS (
                SELECT * FROM {table2_ref} WHERE {where_filter}
            )
            SELECT {pk_select_list('t1')}{non_pk_select_list_both()}
            FROM filtered_t1 t1
            INNER JOIN filtered_t2 t2 ON {join_conditions}
            WHERE {column_diff_conditions()}
            ORDER BY {order_by_list}
            LIMIT {limit}
        """
    else:
        query_diff = f"""
            SELECT {pk_select_list('t1')}{non_pk_select_list_both()}
            FROM {table1_ref} t1
            INNER JOIN {table2_ref} t2 ON {join_conditions}
            WHERE {column_diff_conditions()}
            ORDER BY {order_by_list}
            LIMIT {limit}
        """

    entry = log_request('find_row_differences:diff_values', {
        'table1': f"{project1}.{dataset1}.{table1}",
        'table2': f"{project2}.{dataset2}.{table2}",
        'primary_key': primary_key,
        'where_filter': where_filter
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

    # Add queries for user reference (cleaned up formatting)
    if where_filter:
        results['queries'] = {
            'only_in_source': f"""-- Rows in Source but not in Target (by primary key: {primary_key})
WITH filtered_t1 AS (
    SELECT * FROM {table1_ref} WHERE {where_filter}
),
filtered_t2 AS (
    SELECT * FROM {table2_ref} WHERE {where_filter}
)
SELECT t1.*
FROM filtered_t1 t1
LEFT JOIN filtered_t2 t2 ON {join_conditions}
WHERE {where_null_condition('t2')}""",
            'only_in_target': f"""-- Rows in Target but not in Source (by primary key: {primary_key})
WITH filtered_t1 AS (
    SELECT * FROM {table1_ref} WHERE {where_filter}
),
filtered_t2 AS (
    SELECT * FROM {table2_ref} WHERE {where_filter}
)
SELECT t2.*
FROM filtered_t2 t2
LEFT JOIN filtered_t1 t1 ON {join_conditions}
WHERE {where_null_condition('t1')}""",
            'different_values': f"""-- Rows with different values (by primary key: {primary_key})
-- Note: Excludes _fivetran_id and _fivetran_synced columns from comparison
-- NULL = NULL is treated as equal
WITH filtered_t1 AS (
    SELECT * FROM {table1_ref} WHERE {where_filter}
),
filtered_t2 AS (
    SELECT * FROM {table2_ref} WHERE {where_filter}
)
SELECT {pk_select_list('t1')}{non_pk_select_list_both()}
FROM filtered_t1 t1
INNER JOIN filtered_t2 t2 ON {join_conditions}
WHERE {column_diff_conditions()}
ORDER BY {order_by_list}"""
        }
    else:
        results['queries'] = {
            'only_in_source': f"""-- Rows in Source but not in Target (by primary key: {primary_key})
SELECT t1.*
FROM {table1_ref} t1
LEFT JOIN {table2_ref} t2 ON {join_conditions}
WHERE {where_null_condition('t2')}""",
            'only_in_target': f"""-- Rows in Target but not in Source (by primary key: {primary_key})
SELECT t2.*
FROM {table2_ref} t2
LEFT JOIN {table1_ref} t1 ON {join_conditions}
WHERE {where_null_condition('t1')}""",
            'different_values': f"""-- Rows with different values (by primary key: {primary_key})
-- Note: Excludes _fivetran_id and _fivetran_synced columns from comparison
-- NULL = NULL is treated as equal
SELECT {pk_select_list('t1')}{non_pk_select_list_both()}
FROM {table1_ref} t1
INNER JOIN {table2_ref} t2 ON {join_conditions}
WHERE {column_diff_conditions()}
ORDER BY {order_by_list}"""
        }

    return results


def find_row_differences_no_pk(project1, dataset1, table1, project2, dataset2, table2, common_columns, sample_limit=10, where_filter=None):
    """
    Find rows that exist in one table but not the other using EXCEPT DISTINCT.
    Use this when no primary key is available - compares entire rows.
    Only compares columns that exist in both tables.
    Returns counts and sample rows.
    """
    client = get_client()
    table1_ref = f"`{project1}.{dataset1}.{table1}`"
    table2_ref = f"`{project2}.{dataset2}.{table2}`"

    # Build column list for SELECT (only shared columns)
    columns_sql = ", ".join([f"`{col}`" for col in common_columns])
    
    # Build WHERE clause if filter provided
    where_clause = f" WHERE {where_filter}" if where_filter else ""

    results = {
        'only_in_source_count': 0,
        'only_in_target_count': 0,
        'only_in_source_sample': [],
        'only_in_target_sample': [],
        'queries': {},
        'common_columns_count': len(common_columns)
    }

    # Count rows in source but not in target
    query_count_source = f"""
        SELECT COUNT(*) as cnt FROM (
            SELECT {columns_sql} FROM {table1_ref}{where_clause}
            EXCEPT DISTINCT
            SELECT {columns_sql} FROM {table2_ref}{where_clause}
        )
    """

    entry = log_request('find_row_differences_no_pk:count_only_in_source', {
        'table1': f"{project1}.{dataset1}.{table1}",
        'table2': f"{project2}.{dataset2}.{table2}",
        'where_filter': where_filter
    }, query=query_count_source.strip())
    start = time.time()

    try:
        job = client.query(query_count_source)
        row = list(job.result())[0]
        results['only_in_source_count'] = row.cnt
        log_response(entry, (time.time() - start) * 1000,
                    rows=1, bytes_processed=job.total_bytes_processed)
    except Exception as e:
        results['only_in_source_error'] = str(e)
        log_response(entry, (time.time() - start) * 1000, error=e)

    # Count rows in target but not in source
    query_count_target = f"""
        SELECT COUNT(*) as cnt FROM (
            SELECT {columns_sql} FROM {table2_ref}{where_clause}
            EXCEPT DISTINCT
            SELECT {columns_sql} FROM {table1_ref}{where_clause}
        )
    """

    entry = log_request('find_row_differences_no_pk:count_only_in_target', {
        'table1': f"{project1}.{dataset1}.{table1}",
        'table2': f"{project2}.{dataset2}.{table2}",
        'where_filter': where_filter
    }, query=query_count_target.strip())
    start = time.time()

    try:
        job = client.query(query_count_target)
        row = list(job.result())[0]
        results['only_in_target_count'] = row.cnt
        log_response(entry, (time.time() - start) * 1000,
                    rows=1, bytes_processed=job.total_bytes_processed)
    except Exception as e:
        results['only_in_target_error'] = str(e)
        log_response(entry, (time.time() - start) * 1000, error=e)

    # Sample rows only in source
    query_sample_source = f"""
        SELECT {columns_sql} FROM {table1_ref}{where_clause}
        EXCEPT DISTINCT
        SELECT {columns_sql} FROM {table2_ref}{where_clause}
        LIMIT {sample_limit}
    """

    entry = log_request('find_row_differences_no_pk:sample_only_in_source', {
        'table1': f"{project1}.{dataset1}.{table1}",
        'table2': f"{project2}.{dataset2}.{table2}",
        'limit': sample_limit,
        'where_filter': where_filter
    }, query=query_sample_source.strip())
    start = time.time()

    try:
        job = client.query(query_sample_source)
        for row in job.result():
            results['only_in_source_sample'].append(dict(row))
        log_response(entry, (time.time() - start) * 1000,
                    rows=len(results['only_in_source_sample']),
                    bytes_processed=job.total_bytes_processed)
    except Exception as e:
        results['only_in_source_sample_error'] = str(e)
        log_response(entry, (time.time() - start) * 1000, error=e)

    # Sample rows only in target
    query_sample_target = f"""
        SELECT {columns_sql} FROM {table2_ref}{where_clause}
        EXCEPT DISTINCT
        SELECT {columns_sql} FROM {table1_ref}{where_clause}
        LIMIT {sample_limit}
    """

    entry = log_request('find_row_differences_no_pk:sample_only_in_target', {
        'table1': f"{project1}.{dataset1}.{table1}",
        'table2': f"{project2}.{dataset2}.{table2}",
        'limit': sample_limit,
        'where_filter': where_filter
    }, query=query_sample_target.strip())
    start = time.time()

    try:
        job = client.query(query_sample_target)
        for row in job.result():
            results['only_in_target_sample'].append(dict(row))
        log_response(entry, (time.time() - start) * 1000,
                    rows=len(results['only_in_target_sample']),
                    bytes_processed=job.total_bytes_processed)
    except Exception as e:
        results['only_in_target_sample_error'] = str(e)
        log_response(entry, (time.time() - start) * 1000, error=e)

    # Add queries for user reference (cleaned up formatting)
    results['queries'] = {
        'only_in_source': f"""-- Rows in Source but not in Target (comparing {len(common_columns)} shared columns)
SELECT {columns_sql}
FROM {table1_ref}{where_clause}
EXCEPT DISTINCT
SELECT {columns_sql}
FROM {table2_ref}{where_clause}""",
        'only_in_target': f"""-- Rows in Target but not in Source (comparing {len(common_columns)} shared columns)
SELECT {columns_sql}
FROM {table2_ref}{where_clause}
EXCEPT DISTINCT
SELECT {columns_sql}
FROM {table1_ref}{where_clause}""",
        'sample_only_in_source': f"""-- Sample rows in Source but not in Target (comparing {len(common_columns)} shared columns)
SELECT {columns_sql}
FROM {table1_ref}{where_clause}
EXCEPT DISTINCT
SELECT {columns_sql}
FROM {table2_ref}{where_clause}
LIMIT {sample_limit}""",
        'sample_only_in_target': f"""-- Sample rows in Target but not in Source (comparing {len(common_columns)} shared columns)
SELECT {columns_sql}
FROM {table2_ref}{where_clause}
EXCEPT DISTINCT
SELECT {columns_sql}
FROM {table1_ref}{where_clause}
LIMIT {sample_limit}"""
    }

    return results

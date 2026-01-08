import bigquery_client as bq


def compare_schemas(project1, dataset1, table1, project2, dataset2, table2):
    """
    Compare schemas of two tables.
    Returns detailed comparison with differences highlighted.
    """
    schema1 = bq.get_table_schema(project1, dataset1, table1)
    schema2 = bq.get_table_schema(project2, dataset2, table2)

    # Create lookup dictionaries
    schema1_dict = {col['name']: col for col in schema1}
    schema2_dict = {col['name']: col for col in schema2}

    all_columns = set(schema1_dict.keys()) | set(schema2_dict.keys())

    comparison = []
    has_differences = False

    for col_name in sorted(all_columns):
        col1 = schema1_dict.get(col_name)
        col2 = schema2_dict.get(col_name)

        if col1 is None:
            # Column only in table 2
            comparison.append({
                'column': col_name,
                'table1_type': None,
                'table1_mode': None,
                'table2_type': col2['type'],
                'table2_mode': col2['mode'],
                'status': 'only_in_table2'
            })
            has_differences = True
        elif col2 is None:
            # Column only in table 1
            comparison.append({
                'column': col_name,
                'table1_type': col1['type'],
                'table1_mode': col1['mode'],
                'table2_type': None,
                'table2_mode': None,
                'status': 'only_in_table1'
            })
            has_differences = True
        else:
            # Column in both tables
            type_match = col1['type'] == col2['type']
            mode_match = col1['mode'] == col2['mode']

            if type_match and mode_match:
                status = 'match'
            else:
                status = 'mismatch'
                has_differences = True

            comparison.append({
                'column': col_name,
                'table1_type': col1['type'],
                'table1_mode': col1['mode'],
                'table2_type': col2['type'],
                'table2_mode': col2['mode'],
                'status': status
            })

    return {
        'match': not has_differences,
        'table1_column_count': len(schema1),
        'table2_column_count': len(schema2),
        'details': comparison
    }


def compare_row_counts(project1, dataset1, table1, project2, dataset2, table2):
    """
    Compare row counts of two tables.
    """
    count1 = bq.get_row_count(project1, dataset1, table1)
    count2 = bq.get_row_count(project2, dataset2, table2)

    return {
        'match': count1 == count2,
        'table1_count': count1,
        'table2_count': count2,
        'difference': abs(count1 - count2)
    }


def compare_column_stats(project1, dataset1, table1, project2, dataset2, table2):
    """
    Compare column statistics between two tables.
    Only compares columns that exist in both tables.
    """
    schema1 = bq.get_table_schema(project1, dataset1, table1)
    schema2 = bq.get_table_schema(project2, dataset2, table2)

    # Find common columns
    schema1_dict = {col['name']: col for col in schema1}
    schema2_dict = {col['name']: col for col in schema2}
    common_columns = set(schema1_dict.keys()) & set(schema2_dict.keys())

    # Get stats for common columns only
    common_schema = [col for col in schema1 if col['name'] in common_columns]

    stats1 = bq.get_column_stats(project1, dataset1, table1, common_schema)
    stats2 = bq.get_column_stats(project2, dataset2, table2, common_schema)

    comparison = []
    all_match = True

    for col_name in sorted(common_columns):
        s1 = stats1.get(col_name, {})
        s2 = stats2.get(col_name, {})

        min_match = s1.get('min') == s2.get('min')
        max_match = s1.get('max') == s2.get('max')
        null_match = s1.get('null_count') == s2.get('null_count')
        distinct_match = s1.get('distinct_count') == s2.get('distinct_count')

        col_match = min_match and max_match and null_match and distinct_match
        if not col_match:
            all_match = False

        comparison.append({
            'column': col_name,
            'table1_min': s1.get('min'),
            'table1_max': s1.get('max'),
            'table1_nulls': s1.get('null_count'),
            'table1_distinct': s1.get('distinct_count'),
            'table2_min': s2.get('min'),
            'table2_max': s2.get('max'),
            'table2_nulls': s2.get('null_count'),
            'table2_distinct': s2.get('distinct_count'),
            'match': col_match
        })

    return {
        'match': all_match,
        'details': comparison
    }


def find_mismatches(project1, dataset1, table1, project2, dataset2, table2, primary_key):
    """
    Find sample rows that differ between tables.
    """
    return bq.find_row_differences(project1, dataset1, table1, project2, dataset2, table2, primary_key)


def run_full_comparison(project1, dataset1, table1, project2, dataset2, table2, primary_key=None):
    """
    Run a complete comparison between two tables.
    """
    results = {
        'table1': f"{project1}.{dataset1}.{table1}",
        'table2': f"{project2}.{dataset2}.{table2}",
        'schema': compare_schemas(project1, dataset1, table1, project2, dataset2, table2),
        'row_counts': compare_row_counts(project1, dataset1, table1, project2, dataset2, table2),
        'column_stats': compare_column_stats(project1, dataset1, table1, project2, dataset2, table2)
    }

    # Only run mismatch detection if primary key is provided
    if primary_key:
        results['mismatches'] = find_mismatches(
            project1, dataset1, table1, project2, dataset2, table2, primary_key
        )

    # Overall match status
    results['overall_match'] = (
        results['schema']['match'] and
        results['row_counts']['match'] and
        results['column_stats']['match']
    )

    return results

"""
Unit tests for comparator.py

Run with: pytest tests/test_comparator.py -v
Output saved to: tests/test_output/
"""

import pytest
import json
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comparator

# Output directory for test results
OUTPUT_DIR = Path(__file__).parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_output(filename, data):
    """Save test output as JSON for inspection."""
    filepath = OUTPUT_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nOutput saved to: {filepath}")


# =============================================================================
# FIXTURES - Mock data for testing
# =============================================================================

@pytest.fixture
def schema_table1():
    """Schema for table 1."""
    return [
        {'name': 'id', 'type': 'INT64', 'mode': 'REQUIRED', 'description': ''},
        {'name': 'name', 'type': 'STRING', 'mode': 'NULLABLE', 'description': ''},
        {'name': 'email', 'type': 'STRING', 'mode': 'NULLABLE', 'description': ''},
        {'name': 'created_at', 'type': 'TIMESTAMP', 'mode': 'NULLABLE', 'description': ''},
    ]


@pytest.fixture
def schema_table2_matching():
    """Schema for table 2 - matches table 1."""
    return [
        {'name': 'id', 'type': 'INT64', 'mode': 'REQUIRED', 'description': ''},
        {'name': 'name', 'type': 'STRING', 'mode': 'NULLABLE', 'description': ''},
        {'name': 'email', 'type': 'STRING', 'mode': 'NULLABLE', 'description': ''},
        {'name': 'created_at', 'type': 'TIMESTAMP', 'mode': 'NULLABLE', 'description': ''},
    ]


@pytest.fixture
def schema_table2_different():
    """Schema for table 2 - different from table 1."""
    return [
        {'name': 'id', 'type': 'INT64', 'mode': 'REQUIRED', 'description': ''},
        {'name': 'name', 'type': 'STRING', 'mode': 'REQUIRED', 'description': ''},  # Mode changed
        {'name': 'phone', 'type': 'STRING', 'mode': 'NULLABLE', 'description': ''},  # New column
        {'name': 'created_at', 'type': 'DATE', 'mode': 'NULLABLE', 'description': ''},  # Type changed
        # email column missing
    ]


@pytest.fixture
def stats_table1():
    """Column stats for table 1."""
    return {
        'id': {'min': '1', 'max': '1000', 'null_count': 0, 'distinct_count': 1000},
        'name': {'min': 'Aaron', 'max': 'Zoe', 'null_count': 5, 'distinct_count': 950},
        'email': {'min': 'a@test.com', 'max': 'z@test.com', 'null_count': 10, 'distinct_count': 990},
        'created_at': {'min': '2020-01-01', 'max': '2024-12-31', 'null_count': 0, 'distinct_count': 500},
    }


@pytest.fixture
def stats_table2_matching():
    """Column stats for table 2 - matches table 1."""
    return {
        'id': {'min': '1', 'max': '1000', 'null_count': 0, 'distinct_count': 1000},
        'name': {'min': 'Aaron', 'max': 'Zoe', 'null_count': 5, 'distinct_count': 950},
        'email': {'min': 'a@test.com', 'max': 'z@test.com', 'null_count': 10, 'distinct_count': 990},
        'created_at': {'min': '2020-01-01', 'max': '2024-12-31', 'null_count': 0, 'distinct_count': 500},
    }


@pytest.fixture
def stats_table2_different():
    """Column stats for table 2 - different from table 1."""
    return {
        'id': {'min': '1', 'max': '1000', 'null_count': 0, 'distinct_count': 1000},
        'name': {'min': 'Aaron', 'max': 'Zoe', 'null_count': 10, 'distinct_count': 945},  # Different nulls/distinct
        'email': {'min': 'a@test.com', 'max': 'z@test.com', 'null_count': 10, 'distinct_count': 990},
        'created_at': {'min': '2020-01-01', 'max': '2024-06-30', 'null_count': 0, 'distinct_count': 400},  # Different max/distinct
    }


# =============================================================================
# TESTS - Schema Comparison
# =============================================================================

class TestCompareSchemas:
    """Tests for compare_schemas function."""

    @patch('comparator.bq')
    def test_schemas_match(self, mock_bq, schema_table1, schema_table2_matching):
        """Test when schemas are identical."""
        mock_bq.get_table_schema.side_effect = [schema_table1, schema_table2_matching]

        result = comparator.compare_schemas(
            'proj1', 'dataset1', 'table1',
            'proj2', 'dataset2', 'table2'
        )

        save_output('schema_comparison_match.json', result)

        assert result['match'] is True
        assert result['table1_column_count'] == 4
        assert result['table2_column_count'] == 4
        assert all(col['status'] == 'match' for col in result['details'])

    @patch('comparator.bq')
    def test_schemas_different(self, mock_bq, schema_table1, schema_table2_different):
        """Test when schemas have differences."""
        mock_bq.get_table_schema.side_effect = [schema_table1, schema_table2_different]

        result = comparator.compare_schemas(
            'proj1', 'dataset1', 'table1',
            'proj2', 'dataset2', 'table2'
        )

        save_output('schema_comparison_different.json', result)

        assert result['match'] is False

        # Check specific differences
        details_by_col = {d['column']: d for d in result['details']}

        # 'email' only in table 1
        assert details_by_col['email']['status'] == 'only_in_table1'

        # 'phone' only in table 2
        assert details_by_col['phone']['status'] == 'only_in_table2'

        # 'name' mode mismatch (NULLABLE vs REQUIRED)
        assert details_by_col['name']['status'] == 'mismatch'

        # 'created_at' type mismatch (TIMESTAMP vs DATE)
        assert details_by_col['created_at']['status'] == 'mismatch'

    @patch('comparator.bq')
    def test_empty_schemas(self, mock_bq):
        """Test with empty schemas."""
        mock_bq.get_table_schema.side_effect = [[], []]

        result = comparator.compare_schemas(
            'proj1', 'dataset1', 'table1',
            'proj2', 'dataset2', 'table2'
        )

        save_output('schema_comparison_empty.json', result)

        assert result['match'] is True
        assert result['table1_column_count'] == 0
        assert result['table2_column_count'] == 0


# =============================================================================
# TESTS - Row Count Comparison
# =============================================================================

class TestCompareRowCounts:
    """Tests for compare_row_counts function."""

    @patch('comparator.bq')
    def test_row_counts_match(self, mock_bq):
        """Test when row counts are identical."""
        mock_bq.get_row_count.side_effect = [1000, 1000]

        result = comparator.compare_row_counts(
            'proj1', 'dataset1', 'table1',
            'proj2', 'dataset2', 'table2'
        )

        save_output('row_count_match.json', result)

        assert result['match'] is True
        assert result['table1_count'] == 1000
        assert result['table2_count'] == 1000
        assert result['difference'] == 0

    @patch('comparator.bq')
    def test_row_counts_different(self, mock_bq):
        """Test when row counts differ."""
        mock_bq.get_row_count.side_effect = [1000, 850]

        result = comparator.compare_row_counts(
            'proj1', 'dataset1', 'table1',
            'proj2', 'dataset2', 'table2'
        )

        save_output('row_count_different.json', result)

        assert result['match'] is False
        assert result['table1_count'] == 1000
        assert result['table2_count'] == 850
        assert result['difference'] == 150

    @patch('comparator.bq')
    def test_row_counts_zero(self, mock_bq):
        """Test with empty tables."""
        mock_bq.get_row_count.side_effect = [0, 0]

        result = comparator.compare_row_counts(
            'proj1', 'dataset1', 'table1',
            'proj2', 'dataset2', 'table2'
        )

        save_output('row_count_zero.json', result)

        assert result['match'] is True
        assert result['difference'] == 0


# =============================================================================
# TESTS - Column Statistics Comparison
# =============================================================================

class TestCompareColumnStats:
    """Tests for compare_column_stats function."""

    @patch('comparator.bq')
    def test_stats_match(self, mock_bq, schema_table1, schema_table2_matching,
                         stats_table1, stats_table2_matching):
        """Test when column stats are identical."""
        mock_bq.get_table_schema.side_effect = [schema_table1, schema_table2_matching]
        mock_bq.get_column_stats.side_effect = [stats_table1, stats_table2_matching]

        result = comparator.compare_column_stats(
            'proj1', 'dataset1', 'table1',
            'proj2', 'dataset2', 'table2'
        )

        save_output('column_stats_match.json', result)

        assert result['match'] is True
        assert all(col['match'] for col in result['details'])

    @patch('comparator.bq')
    def test_stats_different(self, mock_bq, schema_table1, schema_table2_matching,
                             stats_table1, stats_table2_different):
        """Test when column stats differ."""
        mock_bq.get_table_schema.side_effect = [schema_table1, schema_table2_matching]
        mock_bq.get_column_stats.side_effect = [stats_table1, stats_table2_different]

        result = comparator.compare_column_stats(
            'proj1', 'dataset1', 'table1',
            'proj2', 'dataset2', 'table2'
        )

        save_output('column_stats_different.json', result)

        assert result['match'] is False

        # Check specific differences
        details_by_col = {d['column']: d for d in result['details']}

        # 'name' has different null_count
        assert details_by_col['name']['match'] is False

        # 'created_at' has different max and distinct_count
        assert details_by_col['created_at']['match'] is False

        # 'id' and 'email' should still match
        assert details_by_col['id']['match'] is True
        assert details_by_col['email']['match'] is True


# =============================================================================
# TESTS - Find Mismatches
# =============================================================================

class TestFindMismatches:
    """Tests for find_mismatches function."""

    @patch('comparator.bq')
    def test_no_mismatches(self, mock_bq):
        """Test when there are no row differences."""
        mock_bq.find_row_differences.return_value = {
            'only_in_table1': [],
            'only_in_table2': [],
            'different_values': []
        }

        result = comparator.find_mismatches(
            'proj1', 'dataset1', 'table1',
            'proj2', 'dataset2', 'table2',
            'id'
        )

        save_output('mismatches_none.json', result)

        assert result['only_in_table1'] == []
        assert result['only_in_table2'] == []
        assert result['different_values'] == []

    @patch('comparator.bq')
    def test_with_mismatches(self, mock_bq):
        """Test when there are row differences."""
        mock_bq.find_row_differences.return_value = {
            'only_in_table1': [
                {'id': 101, 'name': 'John Doe', 'email': 'john@test.com'},
                {'id': 102, 'name': 'Jane Doe', 'email': 'jane@test.com'},
            ],
            'only_in_table2': [
                {'id': 201, 'name': 'New Person', 'email': 'new@test.com'},
            ],
            'different_values': [
                {'id': 50, 'source': 'table1', 'name': 'Old Name'},
                {'id': 50, 'source': 'table2', 'name': 'New Name'},
            ]
        }

        result = comparator.find_mismatches(
            'proj1', 'dataset1', 'table1',
            'proj2', 'dataset2', 'table2',
            'id'
        )

        save_output('mismatches_found.json', result)

        assert len(result['only_in_table1']) == 2
        assert len(result['only_in_table2']) == 1
        assert len(result['different_values']) == 2


# =============================================================================
# TESTS - Full Comparison
# =============================================================================

class TestRunFullComparison:
    """Tests for run_full_comparison function."""

    @patch('comparator.bq')
    def test_full_comparison_match(self, mock_bq, schema_table1, schema_table2_matching,
                                    stats_table1, stats_table2_matching):
        """Test full comparison when tables match."""
        # Setup mocks
        mock_bq.get_table_schema.side_effect = [
            schema_table1, schema_table2_matching,  # For schema comparison
            schema_table1, schema_table2_matching,  # For stats comparison
        ]
        mock_bq.get_row_count.side_effect = [1000, 1000]
        mock_bq.get_column_stats.side_effect = [stats_table1, stats_table2_matching]

        result = comparator.run_full_comparison(
            'proj1', 'dataset1', 'table1',
            'proj2', 'dataset2', 'table2'
        )

        save_output('full_comparison_match.json', result)

        assert result['overall_match'] is True
        assert result['schema']['match'] is True
        assert result['row_counts']['match'] is True
        assert result['column_stats']['match'] is True
        assert 'mismatches' not in result  # No primary key provided

    @patch('comparator.bq')
    def test_full_comparison_different(self, mock_bq, schema_table1, schema_table2_different,
                                        stats_table1, stats_table2_different):
        """Test full comparison when tables differ."""
        # Setup mocks
        mock_bq.get_table_schema.side_effect = [
            schema_table1, schema_table2_different,  # For schema comparison
            schema_table1, schema_table2_different,  # For stats comparison
        ]
        mock_bq.get_row_count.side_effect = [1000, 850]
        mock_bq.get_column_stats.side_effect = [stats_table1, stats_table2_different]

        result = comparator.run_full_comparison(
            'proj1', 'dataset1', 'table1',
            'proj2', 'dataset2', 'table2'
        )

        save_output('full_comparison_different.json', result)

        assert result['overall_match'] is False
        assert result['schema']['match'] is False
        assert result['row_counts']['match'] is False

    @patch('comparator.bq')
    def test_full_comparison_with_primary_key(self, mock_bq, schema_table1, schema_table2_matching,
                                               stats_table1, stats_table2_matching):
        """Test full comparison with primary key (includes mismatch detection)."""
        # Setup mocks
        mock_bq.get_table_schema.side_effect = [
            schema_table1, schema_table2_matching,
            schema_table1, schema_table2_matching,
        ]
        mock_bq.get_row_count.side_effect = [1000, 1000]
        mock_bq.get_column_stats.side_effect = [stats_table1, stats_table2_matching]
        mock_bq.find_row_differences.return_value = {
            'only_in_table1': [],
            'only_in_table2': [],
            'different_values': []
        }

        result = comparator.run_full_comparison(
            'proj1', 'dataset1', 'table1',
            'proj2', 'dataset2', 'table2',
            primary_key='id'
        )

        save_output('full_comparison_with_pk.json', result)

        assert 'mismatches' in result
        assert result['mismatches']['only_in_table1'] == []


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

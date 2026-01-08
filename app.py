from flask import Flask, render_template, jsonify, request
import bigquery_client as bq
import comparator

app = Flask(__name__)


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/projects')
def get_projects():
    """Get list of configured projects."""
    try:
        projects = bq.list_projects()
        return jsonify({'projects': projects})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/<project_id>')
def get_datasets(project_id):
    """Get list of datasets in a project."""
    try:
        datasets = bq.list_datasets(project_id)
        return jsonify({'datasets': datasets})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tables/<project_id>/<dataset_id>')
def get_tables(project_id, dataset_id):
    """Get list of tables in a dataset."""
    try:
        tables = bq.list_tables(project_id, dataset_id)
        return jsonify({'tables': tables})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare', methods=['POST'])
def compare_tables():
    """Run full comparison between two tables."""
    try:
        data = request.get_json()

        project1 = data.get('project1')
        dataset1 = data.get('dataset1')
        table1 = data.get('table1')
        project2 = data.get('project2')
        dataset2 = data.get('dataset2')
        table2 = data.get('table2')
        primary_key = data.get('primary_key')

        if not all([project1, dataset1, table1, project2, dataset2, table2]):
            return jsonify({'error': 'Missing required parameters'}), 400

        results = comparator.run_full_comparison(
            project1, dataset1, table1, project2, dataset2, table2, primary_key
        )

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

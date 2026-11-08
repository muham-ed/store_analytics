import json
from config.database import get_db_cursor

with get_db_cursor() as cursor:
    cursor.execute("SELECT task_id, camera_id, status, started_at, completed_at FROM video_tasks ORDER BY created_at DESC LIMIT 5;")
    tasks = cursor.fetchall()
    print("=== LATEST VIDEO TASKS IN POSTGRESQL ===")
    for t in tasks:
        print(dict(t))

    # Fetch the latest task with non-zero recorded results
    cursor.execute("""
        SELECT scenario_name, metrics_json
        FROM scenario_results
        WHERE task_id = (
            SELECT task_id FROM scenario_results
            WHERE (metrics_json->>'total_in')::int > 0 OR (metrics_json->>'total_footsteps_logged')::float > 0
            ORDER BY result_id DESC LIMIT 1
        );
    """)
    scenarios = cursor.fetchall()
    if not scenarios:
        cursor.execute("SELECT scenario_name, metrics_json FROM scenario_results ORDER BY result_id DESC LIMIT 10;")
        scenarios = cursor.fetchall()

    print(f"\n=== LATEST SCENARIO ANALYTICS RESULTS ({len(scenarios)} Scenarios) ===")
    for s in scenarios:
        print(f"[{s['scenario_name']}]:")
        print(json.dumps(s['metrics_json'], indent=2))

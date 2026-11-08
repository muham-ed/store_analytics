import logging
import json
from typing import Optional, Dict, Any
from config.database import get_db_cursor
from config.settings import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    @staticmethod
    def initialize_db(schema_file_path: str = "database/schema.sql") -> None:
        """
        Executes schema.sql to ensure all tables, indexes, and constraints exist.
        """
        try:
            with open(schema_file_path, "r", encoding="utf-8") as f:
                schema_sql = f.read()
            with get_db_cursor() as cursor:
                cursor.execute(schema_sql)
            logger.info("Database schema initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise e

    @staticmethod
    def fetch_next_task(worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Atomically fetches the next PENDING task, increments attempts, updates heartbeat,
        and locks the row using FOR UPDATE SKIP LOCKED.
        """
        query = """
            WITH next_task AS (
                SELECT task_id
                FROM video_tasks
                WHERE status = 'PENDING' AND attempts < max_attempts
                ORDER BY created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE video_tasks
            SET status = 'PROCESSING',
                locked_by = %s,
                locked_at = NOW(),
                heartbeat_at = NOW(),
                started_at = COALESCE(started_at, NOW()),
                attempts = attempts + 1
            FROM next_task
            WHERE video_tasks.task_id = next_task.task_id
            RETURNING video_tasks.task_id, video_tasks.camera_id, video_tasks.video_path;
        """
        with get_db_cursor() as cursor:
            cursor.execute(query, (worker_id,))
            result = cursor.fetchone()
            if result:
                return dict(result)
        return None

    @staticmethod
    def heartbeat_task(task_id: str) -> None:
        """
        Updates the heartbeat timestamp for an active task.
        """
        query = "UPDATE video_tasks SET heartbeat_at = NOW() WHERE task_id = %s;"
        with get_db_cursor() as cursor:
            cursor.execute(query, (task_id,))

    @staticmethod
    def reap_dead_tasks() -> int:
        """
        Resets tasks whose heartbeat has timed out (> TASK_TIMEOUT_SECONDS) back to PENDING if attempts < max_attempts,
        or marks them FAILED if max_attempts reached.
        """
        query_reap_pending = """
            UPDATE video_tasks
            SET status = 'PENDING',
                locked_by = NULL,
                locked_at = NULL,
                heartbeat_at = NULL
            WHERE status = 'PROCESSING'
              AND heartbeat_at < NOW() - (%s || ' seconds')::INTERVAL
              AND attempts < max_attempts;
        """
        query_reap_failed = """
            UPDATE video_tasks
            SET status = 'FAILED',
                error_message = 'Heartbeat timeout - worker died'
            WHERE status = 'PROCESSING'
              AND heartbeat_at < NOW() - (%s || ' seconds')::INTERVAL
              AND attempts >= max_attempts;
        """
        timeout = settings.TASK_TIMEOUT_SECONDS
        with get_db_cursor() as cursor:
            cursor.execute(query_reap_pending, (str(timeout),))
            reaped_count = cursor.rowcount
            cursor.execute(query_reap_failed, (str(timeout),))
            return reaped_count

    @staticmethod
    def complete_task(task_id: str) -> None:
        """
        Marks a video task as COMPLETED.
        """
        query = """
            UPDATE video_tasks
            SET status = 'COMPLETED',
                completed_at = NOW()
            WHERE task_id = %s;
        """
        with get_db_cursor() as cursor:
            cursor.execute(query, (task_id,))
        logger.info(f"Task {task_id} marked as COMPLETED.")

    @staticmethod
    def fail_task(task_id: str, error_message: str) -> None:
        """
        Marks a video task as FAILED.
        """
        query = """
            UPDATE video_tasks
            SET status = 'FAILED',
                error_message = %s,
                completed_at = NOW()
            WHERE task_id = %s;
        """
        with get_db_cursor() as cursor:
            cursor.execute(query, (error_message, task_id))
        logger.error(f"Task {task_id} marked as FAILED: {error_message}")

    @staticmethod
    def get_camera_config(camera_id: str) -> Dict[str, Any]:
        """
        Retrieves camera configuration geometry.
        """
        query = "SELECT config_json FROM cameras WHERE camera_id = %s;"
        with get_db_cursor() as cursor:
            cursor.execute(query, (camera_id,))
            res = cursor.fetchone()
            if res and res.get("config_json"):
                config = res["config_json"]
                return config if isinstance(config, dict) else json.loads(config)
        return {}

    @staticmethod
    def save_all_scenario_results(task_id: str, camera_id: str, results_dict: Dict[str, Any]) -> None:
        """
        Saves or updates all scenario outputs in a SINGLE atomic database transaction (UPSERT).
        """
        query = """
            INSERT INTO scenario_results (task_id, camera_id, scenario_name, metrics_json)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (task_id, scenario_name)
            DO UPDATE SET
                metrics_json = EXCLUDED.metrics_json,
                timestamp = CURRENT_TIMESTAMP;
        """
        with get_db_cursor() as cursor:
            for scenario_name, metrics in results_dict.items():
                cursor.execute(query, (task_id, camera_id, scenario_name, json.dumps(metrics)))
        logger.info(f"Saved all scenario results for task [{task_id}] in a single transaction.")

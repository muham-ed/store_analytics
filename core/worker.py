import time
import threading
import logging
from config.settings import settings
from database.db_manager import DatabaseManager
from core.detector import RFDetrDetector
from core.tracker import SupervisionTracker
from core.video_processor import VideoProcessor

logger = logging.getLogger(__name__)

class DistributedWorker:
    """
    Distributed Task Worker Daemon:
    Includes heartbeat thread for active task health monitoring,
    reaper thread for recovering stalled/dead worker tasks,
    and atomic task claim execution.
    """
    def __init__(self):
        self.worker_id = settings.WORKER_ID
        logger.info(f"Initializing Distributed Worker Daemon: {self.worker_id}")

        self.detector = RFDetrDetector()
        self.tracker = SupervisionTracker()
        self.processor = VideoProcessor(detector=self.detector, tracker=self.tracker)
        self._stop_event = threading.Event()

    def _start_reaper_thread(self):
        """Background thread to periodically reap dead worker tasks."""
        def reaper_loop():
            while not self._stop_event.is_set():
                try:
                    reaped = DatabaseManager.reap_dead_tasks()
                    if reaped > 0:
                        logger.warning(f"Reaper recovered {reaped} stalled task(s).")
                except Exception as err:
                    logger.error(f"Reaper thread exception: {err}")
                self._stop_event.wait(settings.REAPER_INTERVAL_SECONDS)

        t = threading.Thread(target=reaper_loop, daemon=True, name="ReaperThread")
        t.start()

    def start(self):
        logger.info(f"Worker {self.worker_id} polling for tasks...")

        try:
            DatabaseManager.initialize_db()
        except Exception as e:
            logger.warning(f"DB Schema check: {e}")

        self._start_reaper_thread()

        while not self._stop_event.is_set():
            try:
                task = DatabaseManager.fetch_next_task(worker_id=self.worker_id)
                if task:
                    task_id = str(task["task_id"])
                    camera_id = task["camera_id"]
                    video_path = task["video_path"]

                    logger.info(f"Worker {self.worker_id} claimed Task [{task_id}]")

                    # Heartbeat Thread for Active Task
                    task_stop_hb = threading.Event()
                    def heartbeat_loop():
                        while not task_stop_hb.is_set():
                            try:
                                DatabaseManager.heartbeat_task(task_id)
                            except Exception as hb_err:
                                logger.error(f"Heartbeat error for task [{task_id}]: {hb_err}")
                            task_stop_hb.wait(settings.HEARTBEAT_INTERVAL_SECONDS)

                    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
                    hb_thread.start()

                    try:
                        self.processor.process_video_task(
                            task_id=task_id,
                            camera_id=camera_id,
                            video_path=video_path
                        )
                        DatabaseManager.complete_task(task_id=task_id)
                    except Exception as exec_err:
                        err_msg = str(exec_err)
                        logger.error(f"Task [{task_id}] failed: {err_msg}")
                        DatabaseManager.fail_task(task_id=task_id, error_message=err_msg)
                    finally:
                        task_stop_hb.set()
                        hb_thread.join(timeout=2.0)

                else:
                    time.sleep(settings.POLL_INTERVAL_SECONDS)

            except KeyboardInterrupt:
                logger.info("Worker shutting down.")
                self._stop_event.set()
                break
            except Exception as loop_err:
                logger.error(f"Worker loop exception: {loop_err}")
                time.sleep(settings.POLL_INTERVAL_SECONDS)

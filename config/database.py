import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

def ensure_database_exists():
    """
    Ensures the target database (settings.POSTGRES_DB) exists, creating it if necessary.
    """
    try:
        conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            dbname="postgres",
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD
        )
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (settings.POSTGRES_DB,))
        exists = cursor.fetchone()
        if not exists:
            cursor.execute(f'CREATE DATABASE "{settings.POSTGRES_DB}"')
            logger.info(f"Database '{settings.POSTGRES_DB}' created successfully.")
        cursor.close()
        conn.close()
    except Exception as e:
        logger.warning(f"Could not verify database existence: {e}")

def get_db_connection():
    """
    Returns a new PostgreSQL database connection using psycopg2.
    """
    ensure_database_exists()
    return psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD
    )

@contextmanager
def get_db_cursor():
    """
    Context manager for safely executing DB queries with auto-commit and rollback.
    """
    conn = get_db_connection()
    conn.autocommit = False
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database transaction error: {str(e)}")
        raise e
    finally:
        cursor.close()
        conn.close()

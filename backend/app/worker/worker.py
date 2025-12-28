"""
RQ Worker for Saree Virtual Try-On.

Consumes jobs from Redis queue and runs the pipeline orchestrator.
"""

import logging
import os
import sys

from redis import Redis
from rq import Worker, Queue, Connection

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(job_id)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Add custom filter to include job_id in logs
class JobContextFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'job_id'):
            record.job_id = 'worker'
        return True

logger = logging.getLogger(__name__)
logger.addFilter(JobContextFilter())


def run_worker():
    """Start the RQ worker to process pipeline jobs."""
    logger.info(f"Starting worker, connecting to Redis: {settings.REDIS_URL}")
    
    try:
        redis_conn = Redis.from_url(settings.REDIS_URL)
        
        # Test connection
        redis_conn.ping()
        logger.info("Connected to Redis successfully")
        
        with Connection(redis_conn):
            worker = Worker(
                queues=[settings.RQ_QUEUE_NAME],
                name=f"saree-worker-{os.getpid()}",
            )
            
            logger.info(f"Worker ready, listening on queue: {settings.RQ_QUEUE_NAME}")
            worker.work(with_scheduler=False)
            
    except Exception as e:
        logger.error(f"Worker failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_worker()

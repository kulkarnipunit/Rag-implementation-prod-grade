"""
Async job queue with concurrent workers, crash recovery, and webhook delivery.

Crash recovery model:
  - Every job tracks processing_started_at when a worker picks it up
  - A reaper coroutine scans every REAPER_INTERVAL_SECONDS
  - Any job stuck in "processing" for > JOB_TIMEOUT_SECONDS is assumed crashed
  - The reaper re-enqueues it (up to MAX_RETRIES times), then marks it failed

Why re-running from scratch is safe (idempotency):
  - load_documents → pure read, safe to repeat
  - chunk_documents → pure function, same output for same input
  - embed_chunks → deterministic, same vectors
  - store.add_chunks → ChromaDB upserts by ID, duplicates are overwritten not doubled

The pattern (mark-then-timeout) is exactly how SQS VisibilityTimeout works in prod:
  - SQS hides a message for N seconds when a worker receives it
  - If the worker doesn't delete it in time, SQS makes it visible again → another worker picks it up
  - We replicate that here in-process with the reaper task
"""
import asyncio
import concurrent.futures
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

import httpx

WORKER_CONCURRENCY    = int(os.getenv("WORKER_CONCURRENCY", "3"))
JOB_TIMEOUT_SECONDS   = int(os.getenv("JOB_TIMEOUT_SECONDS", "300"))   # 5 min
MAX_RETRIES           = int(os.getenv("MAX_RETRIES", "3"))
REAPER_INTERVAL       = int(os.getenv("REAPER_INTERVAL_SECONDS", "30"))


class Job:
    def __init__(self, payload: Dict[str, Any], webhook_url: Optional[str] = None):
        self.job_id: str = str(uuid.uuid4())
        self.payload = payload
        self.webhook_url = webhook_url
        self.status: str = "queued"
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.retry_count: int = 0
        self.created_at: str = _now()
        self.updated_at: str = self.created_at
        self.processing_started_at: Optional[str] = None  # set when worker picks up

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "payload": self.payload,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "processing_started_at": self.processing_started_at,
        }

    def _touch(self):
        self.updated_at = _now()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seconds_since(iso_str: str) -> float:
    dt = datetime.fromisoformat(iso_str)
    return (datetime.now(timezone.utc) - dt).total_seconds()


ProcessorFn = Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]


class JobManager:
    """
    In-process async job queue with N concurrent workers and crash recovery.

    Recovery mechanism:
      The reaper task wakes every REAPER_INTERVAL seconds.
      Any job with status="processing" whose processing_started_at is older than
      JOB_TIMEOUT_SECONDS is considered crashed. It is re-enqueued with retry_count+1.
      After MAX_RETRIES failed attempts the job is marked "failed" permanently.
    """

    def __init__(self, concurrency: int = WORKER_CONCURRENCY):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._jobs: Dict[str, Job] = {}
        self._processor: Optional[ProcessorFn] = None
        self._worker_tasks: List[asyncio.Task] = []
        self._reaper_task: Optional[asyncio.Task] = None
        self._concurrency = concurrency
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="rag-worker"
        )

    def set_processor(self, fn: ProcessorFn):
        self._processor = fn

    @property
    def executor(self) -> concurrent.futures.ThreadPoolExecutor:
        return self._executor

    def enqueue(self, payload: Dict[str, Any], webhook_url: Optional[str] = None) -> Job:
        job = Job(payload=payload, webhook_url=webhook_url)
        self._jobs[job.job_id] = job
        self._queue.put_nowait(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def stats(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {"queued": 0, "processing": 0, "success": 0, "failed": 0}
        for job in self._jobs.values():
            counts[job.status] = counts.get(job.status, 0) + 1
        retried = sum(1 for j in self._jobs.values() if j.retry_count > 0)
        return {
            "concurrency": self._concurrency,
            "queue_depth": self._queue.qsize(),
            "total_jobs": len(self._jobs),
            "retried_jobs": retried,
            "job_timeout_seconds": JOB_TIMEOUT_SECONDS,
            "max_retries": MAX_RETRIES,
            **counts,
        }

    def start_worker(self):
        loop = asyncio.get_event_loop()
        for i in range(self._concurrency):
            task = loop.create_task(self._worker(), name=f"rag-worker-{i}")
            self._worker_tasks.append(task)
        self._reaper_task = loop.create_task(self._reaper(), name="rag-reaper")

    async def stop_worker(self):
        all_tasks = self._worker_tasks + ([self._reaper_task] if self._reaper_task else [])
        for task in all_tasks:
            task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        self._reaper_task = None
        self._executor.shutdown(wait=False)

    # -------------------------------------------------------------------------
    # Worker
    # -------------------------------------------------------------------------

    async def _worker(self):
        while True:
            job: Job = await self._queue.get()
            job.status = "processing"
            job.processing_started_at = _now()   # reaper uses this to detect crashes
            job._touch()
            try:
                result = await self._processor(job.payload)
                job.status = "success"
                job.result = result
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
            finally:
                job._touch()
                self._queue.task_done()
                await self._fire_webhook(job)

    # -------------------------------------------------------------------------
    # Reaper — detects crashed jobs and re-enqueues them
    # -------------------------------------------------------------------------

    async def _reaper(self):
        """
        Runs every REAPER_INTERVAL seconds.
        Finds jobs stuck in "processing" longer than JOB_TIMEOUT_SECONDS.
        Re-enqueues them if retries remain, else marks them permanently failed.
        """
        while True:
            await asyncio.sleep(REAPER_INTERVAL)
            for job in list(self._jobs.values()):
                if job.status != "processing":
                    continue
                if not job.processing_started_at:
                    continue
                elapsed = _seconds_since(job.processing_started_at)
                if elapsed < JOB_TIMEOUT_SECONDS:
                    continue

                # Job is stuck — decide: retry or give up
                if job.retry_count < MAX_RETRIES:
                    job.retry_count += 1
                    job.status = "queued"
                    job.processing_started_at = None
                    job.error = f"Worker timeout after {elapsed:.0f}s — retry {job.retry_count}/{MAX_RETRIES}"
                    job._touch()
                    self._queue.put_nowait(job)
                else:
                    job.status = "failed"
                    job.error = f"Permanently failed after {MAX_RETRIES} retries (last timeout: {elapsed:.0f}s)"
                    job._touch()
                    await self._fire_webhook(job)

    # -------------------------------------------------------------------------
    # Webhook
    # -------------------------------------------------------------------------

    async def _fire_webhook(self, job: Job):
        if not job.webhook_url:
            return
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(job.webhook_url, json=job.to_dict())
        except Exception:
            pass


# Module-level singleton imported by the API layer
manager = JobManager()

"""Runs inside the sandbox container. Re-checks the code, then executes it in a child process with a timeout."""
from fastapi import FastAPI
from pydantic import BaseModel

from sandbox import _run_local, check  # same static checks + runner as the API side

app = FastAPI()


class Job(BaseModel):
    code: str
    records: list[dict] = []
    calls: list[dict] = []
    zones: list[str] = []


@app.post("/run")
def run(job: Job):
    err = check(job.code)
    if err:
        return {"ok": False, "error": err}
    return _run_local(job.code, job.records, job.calls, job.zones)

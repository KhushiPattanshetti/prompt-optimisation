"""
Stress / load tests using Locust.

Simulates concurrent requests against the RL microservice API.

Scenario:
    - 1000 rollout batches are pre-written to a temp directory
    - Virtual users hit /status and /train concurrently
    - Measures API latency and throughput

Run with:
    locust -f tests/stress/locustfile.py --host http://localhost:8000 \
           --users 10 --spawn-rate 2 --run-time 60s --headless

Or launch the Locust web UI:
    locust -f tests/stress/locustfile.py --host http://localhost:8000
"""

import random
from locust import HttpUser, between, task


class RLServiceUser(HttpUser):
    """Simulates a client polling and triggering the RL training service."""

    wait_time = between(0.1, 0.5)

    @task(5)
    def get_status(self):
        """Heavily poll the /status endpoint (higher weight)."""
        with self.client.get("/status", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 503:
                resp.failure("Service not initialised")
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

    @task(2)
    def trigger_train(self):
        """Trigger a training cycle via POST /train."""
        with self.client.post("/train", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 503:
                resp.failure("Service not initialised")
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

    @task(1)
    def get_checkpoint(self):
        """Fetch latest checkpoint metadata."""
        with self.client.get("/checkpoint", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 503:
                resp.failure("Service not initialised")
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

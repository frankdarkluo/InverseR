import base64
import json
import logging
import sqlite3
from pathlib import Path

import httpx
from pydantic import Field

from inverser.metrics import EndpointScore, score_endpoint
from inverser.model_adapters import ModelAdapter, ModelCall, ModelReply
from inverser.physics import SnookerSimulator
from inverser.runtime import fingerprint, write_json
from inverser.schemas import CueAction, Record, ShotRecord
from inverser.study import GoalSubmission, SubmittedAction, make_study_case

LOGGER = logging.getLogger(__name__)


class BudgetExhausted(RuntimeError):
    pass


class UnresolvedRequest(RuntimeError):
    pass


class BudgetLimits(Record):
    max_model_requests: int = Field(ge=0)
    max_physics_evaluations: int = Field(ge=0)
    max_physics_attempts: int | None = Field(default=None, ge=0)


class TrialLedger:
    def __init__(self, path: Path, limits: BudgetLimits) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.limits = limits
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests (
              request_key TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              reply_json TEXT,
              error_code TEXT
            );
            CREATE TABLE IF NOT EXISTS physics (
              id INTEGER PRIMARY KEY,
              status TEXT NOT NULL,
              evaluations INTEGER
            );
            """
        )
        expected = fingerprint(limits.model_dump(mode="json"))
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key='budget_hash'"
        ).fetchone()
        if row is None:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO metadata(key,value) VALUES ('budget_hash',?)", (expected,)
                )
        elif row[0] != expected:
            self.connection.close()
            raise ValueError("Changing a run's budget requires an explicitly new ledger")

    def close(self) -> None:
        self.connection.close()

    def counts(self) -> dict[str, int]:
        return {
            "model_requests": self.connection.execute("SELECT COUNT(*) FROM requests").fetchone()[
                0
            ],
            "physics_attempts": self.connection.execute("SELECT COUNT(*) FROM physics").fetchone()[
                0
            ],
            "physics_evaluations": self.connection.execute(
                "SELECT COALESCE(SUM(evaluations),0) FROM physics"
            ).fetchone()[0],
        }

    def reserve_request(self, key: str) -> ModelReply | None:
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            row = self.connection.execute(
                "SELECT status,reply_json FROM requests WHERE request_key=?", (key,)
            ).fetchone()
            if row is not None:
                if row[0] == "complete":
                    return ModelReply.model_validate_json(row[1]).model_copy(
                        update={"cached": True}
                    )
                raise UnresolvedRequest(
                    "Existing failed/unknown request must not be automatically resubmitted"
                )
            if self.counts()["model_requests"] >= self.limits.max_model_requests:
                raise BudgetExhausted("Model request budget exhausted")
            self.connection.execute(
                "INSERT INTO requests(request_key,status) VALUES (?,'reserved')", (key,)
            )
        return None

    def finish_request(self, key: str, reply: ModelReply) -> None:
        with self.connection:
            changed = self.connection.execute(
                "UPDATE requests SET status='complete',reply_json=? "
                "WHERE request_key=? AND status='reserved'",
                (reply.model_dump_json(), key),
            ).rowcount
            if changed != 1:
                raise ValueError("A reply must finish exactly one reserved request")

    def fail_request(self, key: str, code: str, *, unknown_response: bool) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE requests SET status=?,error_code=? "
                "WHERE request_key=? AND status='reserved'",
                ("unknown_response" if unknown_response else "known_error", code, key),
            )
        LOGGER.error("Model request %s failed: %s", key, code)

    def reserve_physics(self) -> int:
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            occupied = (
                self.connection.execute(
                    "SELECT COALESCE(SUM(evaluations),0) + "
                    "SUM(CASE WHEN status='reserved' THEN 1 ELSE 0 END) FROM physics"
                ).fetchone()[0]
                or 0
            )
            attempt_limit = self.limits.max_physics_attempts
            if attempt_limit is None:
                attempt_limit = self.limits.max_physics_evaluations
            if (
                occupied >= self.limits.max_physics_evaluations
                or self.counts()["physics_attempts"] >= attempt_limit
            ):
                raise BudgetExhausted("Physics evaluation/attempt budget exhausted")
            cursor = self.connection.execute("INSERT INTO physics(status) VALUES ('reserved')")
            if cursor.lastrowid is None:
                raise RuntimeError("Physics reservation was not persisted")
            return cursor.lastrowid

    def record_physics(self, reservation: int, evaluations: int) -> None:
        if evaluations not in (0, 1):
            raise ValueError("An instrumented replay must execute zero or one forward evaluations")
        with self.connection:
            changed = self.connection.execute(
                "UPDATE physics SET status='complete',evaluations=? "
                "WHERE id=? AND status='reserved'",
                (evaluations, reservation),
            ).rowcount
            if changed != 1:
                raise ValueError("Physics completion must match one reserved attempt")

    def usage_totals(self) -> dict[str, int | None]:
        replies = [
            ModelReply.model_validate_json(row[0])
            for row in self.connection.execute(
                "SELECT reply_json FROM requests WHERE status='complete'"
            )
        ]
        fields = {
            "input_tokens": [reply.usage.input_tokens for reply in replies],
            "output_tokens": [reply.usage.output_tokens for reply in replies],
            "total_tokens": [reply.usage.total_tokens for reply in replies],
        }
        return {
            field: sum(value for value in values if value is not None)
            if all(value is not None for value in values)
            else None
            for field, values in fields.items()
        }


def request_model(adapter: ModelAdapter, call: ModelCall, ledger: TrialLedger) -> ModelReply:
    key = adapter.request_key(call)
    cached = ledger.reserve_request(key)
    if cached is not None:
        return cached
    try:
        reply = adapter._complete(call, key)
    except httpx.RequestError:
        ledger.fail_request(key, "transport_response_unknown", unknown_response=True)
        raise
    except httpx.HTTPStatusError:
        ledger.fail_request(key, "http_status_error", unknown_response=False)
        raise
    except ValueError:
        ledger.fail_request(key, "response_identity_format_or_usage_error", unknown_response=False)
        raise
    ledger.finish_request(key, reply)
    return reply


def evaluate_submission(
    record: ShotRecord, action: CueAction, simulator: SnookerSimulator, ledger: TrialLedger
) -> EndpointScore:
    if simulator.config != record.physics:
        raise ValueError("Reference simulator physics must match the case")
    reservation = ledger.reserve_physics()
    before = simulator.rollouts
    try:
        replay = simulator.replay(record.initial, action)
    finally:
        ledger.record_physics(reservation, simulator.rollouts - before)
    return score_endpoint(
        replay.goal,
        record.goal,
        record.physics.ball_radius_m,
        no_scratch=True,
    )


def study_dry_run(source: Path, output: Path) -> dict[str, object]:
    with source.open() as stream:
        line = stream.readline()
    if not line:
        raise ValueError("Dry run requires an existing simulator development record")
    record = ShotRecord.model_validate_json(line)
    simulator = SnookerSimulator(record.physics)
    case = make_study_case(record, simulator)
    submission = GoalSubmission(
        action=SubmittedAction.model_validate(record.witness_action.model_dump(mode="json"))
    )

    def fixture_response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "fixture-only",
                "choices": [{"message": {"content": submission.model_dump_json()}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

    from inverser.model_adapters import BackendConfig, rendered_asset

    output.mkdir(parents=True, exist_ok=False)
    ledger = TrialLedger(
        output / "ledger.sqlite",
        BudgetLimits(max_model_requests=2, max_physics_evaluations=2),
    )
    try:
        with ModelAdapter(
            BackendConfig(
                kind="mock",
                alias="mock-engineering-fixture",
                model_id="fixture-only",
                model_revision="fixture-v1",
                base_url="https://fixture.invalid/v1",
            ),
            transport=httpx.MockTransport(fixture_response),
        ) as adapter:
            goal = case.goal_input()
            results = []
            for view in ("structured", "bev"):
                images = (
                    (
                        rendered_asset(goal, "initial", goal.physics.table),
                        rendered_asset(goal, "goal", goal.physics.table),
                    )
                    if view == "bev"
                    else ()
                )
                call = ModelCall(view=view, input=goal, images=images)
                reply = request_model(adapter, call, ledger)
                cached = request_model(adapter, call, ledger)
                score = evaluate_submission(
                    record,
                    GoalSubmission.model_validate_json(reply.content).action,
                    simulator,
                    ledger,
                )
                results.append(
                    {
                        "view": view,
                        "success": score.success,
                        "max_position_error_m": score.max_position_error_m,
                        "second_request_cached": cached.cached,
                    }
                )
                write_json(output / f"{view}_input.json", json.loads(call.prompt()))
                for image in images:
                    (output / f"{image.role}.png").write_bytes(base64.b64decode(image.png_base64))
            report: dict[str, object] = {
                "status": "engineering_mock_dry_run",
                "actual_model_inference": False,
                "paid_api_requests": 0,
                "fixture_action_source": "privileged_witness_for_engineering_only",
                "model_performance_claim": False,
                "case_id": case.case_id,
                "results": results,
                "counts": ledger.counts(),
                "usage": ledger.usage_totals(),
            }
            write_json(output / "validation.json", report)
            return report
    finally:
        ledger.close()

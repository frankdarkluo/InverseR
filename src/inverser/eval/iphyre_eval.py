"""Ledgered DeepPHY I-PHYRE adaptation using the pinned isolated physics runner."""

import base64
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import httpx
from pydantic import Field, StrictInt

from inverser.evaluation import BudgetExhausted, BudgetLimits, TrialLedger, UnresolvedRequest
from inverser.runtime import PROJECT_ROOT, file_sha256, fingerprint, write_json
from inverser.schemas import Record
from inverser.study_calls import StudyAdapter, StudyBackendConfig, load_job, lock_run, save_job
from inverser.vlm_eval import _png_asset, visual_request

ATTEMPTS = 10
SUITES = {"iphyre": "02_iphyre", "phyre": "01_phyre", "pooltool": "04_pooltool"}


class Elimination(Record):
    index: StrictInt = Field(ge=0)
    time: float = Field(ge=0, le=15)


class EliminationPlan(Record):
    prediction: str = Field(max_length=800)
    actions: tuple[Elimination, ...] = Field(max_length=16)


class PlacementPlan(Record):
    prediction: str = Field(max_length=800)
    cell: StrictInt = Field(ge=1, le=64)
    radius: StrictInt = Field(ge=1, le=8)


class PoolPlan(Record):
    prediction: str = Field(max_length=800)
    speed: str
    strikespot: str


ANSWER_SCHEMA: dict[str, type[Record]] = {
    "iphyre": EliminationPlan,
    "phyre": PlacementPlan,
    "pooltool": PoolPlan,
}


def iphyre_call(request: dict, suite: str = "iphyre") -> dict[str, Any]:
    if suite not in SUITES:
        raise ValueError("Unknown discrete physics protocol")
    result = subprocess.run(
        [
            str(
                PROJECT_ROOT
                / ("envs/main/bin/python" if suite == "pooltool" else f"envs/{suite}/bin/python")
            ),
            str(PROJECT_ROOT / f"scripts/{suite}_runner.py"),
        ],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
        env=dict(os.environ, PYTHONHASHSEED="0"),
    )
    return json.loads(result.stdout)


def run_iphyre(
    output: Path,
    backend: StudyBackendConfig,
    *,
    budget_tokens: int,
    execute: bool,
    count: int | None,
    suite: str = "iphyre",
    transport: httpx.BaseTransport | None = None,
) -> dict[str, object]:
    if count is not None and count <= 0:
        raise ValueError("The scene limit must be positive")
    if suite not in SUITES:
        raise ValueError("Unknown discrete physics protocol")
    runtime = PROJECT_ROOT / f"data/external/verification-{suite}-v1.json"
    proof = json.loads(runtime.read_text())
    if proof["status"] != "passed" or proof["runner_sha256"] != file_sha256(
        PROJECT_ROOT / f"scripts/{suite}_runner.py"
    ):
        raise ValueError("The isolated I-PHYRE runtime has not been verified")
    config = PROJECT_ROOT / f"data/external/DeepPHY/conf/env_config_{SUITES[suite]}.json"
    settings = json.loads(config.read_text())
    attempts = settings["max_attempts_per_game"] if suite == "pooltool" else ATTEMPTS
    if suite == "iphyre":
        patch = PROJECT_ROOT / "data/external/DeepPHY/res/02_iphyre/simulator.py"
        if proof["patch_sha256"] != file_sha256(patch):
            raise ValueError("The verified I-PHYRE simulator patch changed")
    elif proof["config_sha256"] != file_sha256(config):
        raise ValueError("The verified PHYRE grid configuration changed")
    if suite != "pooltool" and settings["max_attempts"] != ATTEMPTS:
        raise ValueError("DeepPHY attempt budget changed")
    tasks = iphyre_call({"op": "list"}, suite)["tasks"][:count]
    limits = BudgetLimits(
        max_model_requests=len(tasks) * attempts,
        max_physics_evaluations=len(tasks) * attempts,
        max_total_tokens=budget_tokens,
    )
    run_hash = lock_run(
        output,
        {
            "schema_version": f"deepphy-{suite}-adaptation-v1",
            "task_names": tasks,
            "runtime_receipt": file_sha256(runtime),
            "upstream_config": file_sha256(config),
            "backend": backend.model_dump(mode="json"),
            "endpoint": fingerprint(backend.base_url),
            "limits": limits.model_dump(mode="json"),
            "protocol": (
                "nine_ball_continuing_state_top_speed_and_scratches_logged"
                if suite == "pooltool"
                else "pinned_physics_scorer_WM_prediction_scalar_feedback_initial_PNG_only"
            ),
            "producer": file_sha256(Path(__file__)),
        },
        reserve_bytes=20_000_000,
    )
    if not execute:
        return {"status": "planned_no_calls", "request_cap": limits.max_model_requests}
    ledger = TrialLedger(output / "ledger.sqlite", limits)
    completed: list[dict[str, Any]] = []
    stop = None
    try:
        with StudyAdapter(
            backend.model_copy(update={"execution_authorized": True}), transport=transport
        ) as client:
            for index, game in enumerate(tasks):
                job = output / "jobs" / f"{index:04d}.json"
                if (saved := load_job(job, run_hash)) is not None:
                    completed.append(saved)
                    continue
                initial = iphyre_call({"op": "render", "game": game}, suite)
                png = _png_asset(base64.b64decode(initial["png"], validate=True), game)
                if png.sha256 != initial["png_sha256"]:
                    raise ValueError("Rendered IPC image checksum mismatch")
                history = []
                blocks = int(initial.get("eliminable_blocks", 0))
                state = initial.get("state")
                for turn in range(1, attempts + 1):
                    attempt_file = output / "turns" / f"{index:04d}-{turn:02d}.json"
                    if (saved := load_job(attempt_file, run_hash)) is not None:
                        history.append(saved)
                        state = saved.get("state", state)
                        if saved["solved"]:
                            break
                        continue
                    prompt = (
                        "I-PHYRE: make every red ball fall into the abyss by removing selected "
                        f"numbered gray blocks. There are {blocks} removable blocks, "
                        "numbered from 0. Black structures are fixed. Blue blocks are dynamic. "
                        "Predict the motion, then select eliminations at 0–15 seconds. Use "
                        'JSON {"prediction":"...", "actions":[{"index":0,"time":0.5}]}. '
                        "Earlier attempts restarted from the same scene. Use their actual "
                        "reward/outcome to revise the next attempt."
                    )
                    if suite == "phyre":
                        prompt = (
                            "PHYRE: use a red ball to make the green object touch the blue "
                            "object (or the goal objects shown in the scene). Pick a grid cell "
                            "1–64 and ball radius level 1–8. Cells are numbered left to right, "
                            "top to bottom. Predict the result of your choice. Return JSON "
                            '{"prediction":"...", "cell":12,"radius":2}. Prior actions restart '
                            "from the same scene; use their actual status to revise."
                        )
                    elif suite == "pooltool":
                        prompt = (
                            "Nine-ball billiards. Pocket ball 9 by shooting toward the lowest "
                            "remaining numbered ball; avoid scratching ball 0 (cue). The simulator "
                            "aims at that object automatically. Predict the resulting events, then "
                            f"choose speed from {list(settings['speed_mapping'])} and strikespot "
                            f"from {list(settings['strikespot_mapping'])}. "
                            'JSON: {"prediction":"...", "speed":"Low", "strikespot":"Middle Spin"}.'
                        )
                        if turn > 1:
                            current = iphyre_call(
                                {"op": "render", "game": game, "state": state}, suite
                            )
                            png = _png_asset(base64.b64decode(current["png"], validate=True), game)
                            if png.sha256 != current["png_sha256"]:
                                raise ValueError("Current nine-ball PNG checksum mismatch")
                    if history:
                        prompt += "\nPrior attempts:\n" + "\n".join(
                            json.dumps(
                                {key: value for key, value in item.items() if key != "reply"}
                            )
                            for item in history
                        )
                    reply = visual_request(
                        client,
                        ledger,
                        prompt,
                        (png,),
                        episode_key=f"{suite}/{game}/{turn}",
                        seed=0,
                        output_schema=ANSWER_SCHEMA[suite].model_json_schema(),
                    )
                    outcome: dict[str, Any]
                    try:
                        if suite == "iphyre":
                            plan = EliminationPlan.model_validate_json(reply.content)
                            if any(action.index >= blocks for action in plan.actions):
                                raise ValueError("Elimination index outside action space")
                            proposal = plan.model_dump(mode="json")
                        elif suite == "phyre":
                            proposal = PlacementPlan.model_validate_json(reply.content).model_dump(
                                mode="json"
                            )
                        else:
                            shot = PoolPlan.model_validate_json(reply.content)
                            if (
                                shot.speed not in settings["speed_mapping"]
                                or shot.strikespot not in settings["strikespot_mapping"]
                            ):
                                raise ValueError("Invalid nine-ball named control")
                            proposal = shot.model_dump(mode="json")
                    except ValueError:
                        outcome = {"solved": False, "reward": None, "invalid": True}
                    else:
                        reservation = ledger.reserve_physics()
                        try:
                            outcome = iphyre_call(
                                {"op": "simulate", "game": game, "state": state, **proposal}, suite
                            )
                        except (subprocess.SubprocessError, ValueError):
                            ledger.record_physics(reservation, 0)
                            raise
                        ledger.record_physics(reservation, 1)
                        outcome |= proposal
                        state = outcome.get("state", state)
                    outcome |= {"turn": turn, "reply": reply.model_dump(mode="json")}
                    save_job(attempt_file, run_hash, outcome)
                    history.append(outcome)
                    if outcome["solved"]:
                        break
                row = {"game": game, "solved": bool(history[-1]["solved"]), "attempts": history}
                save_job(job, run_hash, row)
                completed.append(row)
    except (
        httpx.HTTPError,
        BudgetExhausted,
        UnresolvedRequest,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        stop = type(error).__name__
    finally:
        report: dict[str, object] = {
            "status": "complete" if stop is None and len(completed) == len(tasks) else "stopped",
            "run_hash": run_hash,
            "jobs": len(completed),
            "stop_error": stop,
            "solved": sum(row["solved"] for row in completed),
            "nine_ball_diagnostics": (
                {
                    "top_speed_shots": sum(
                        bool(item.get("top_speed")) for row in completed for item in row["attempts"]
                    ),
                    "scratches": sum(
                        "cue" in item.get("potted", [])
                        for row in completed
                        for item in row["attempts"]
                    ),
                }
                if suite == "pooltool"
                else None
            ),
            "counts": ledger.counts(),
            "usage": ledger.usage_totals(),
            "request_status": dict(
                ledger.connection.execute("SELECT status, COUNT(*) FROM requests GROUP BY status")
            ),
        }
        write_json(output / "summary.json", report)
        ledger.close()
    return report

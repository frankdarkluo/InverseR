"""Reproduce the observed analysis from hashed job envelopes without model or physics calls."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from inverser.capo_figures import write_figures
from inverser.diagnostics import PairedMeasurement, summarize_paired
from inverser.runtime import file_sha256, fingerprint, write_json
from inverser.study_calls import create_artifact_dir, load_job

FORWARD_SUMMARY_KEYS = (
    "T1",
    "T2",
    "T3",
    "S_total",
    "first_contact_exact",
    "cushion_sequence_exact",
    "T1_per_ball",
    "T2_cushion_set",
    "T3_1r",
    "T3_2r",
    "T3_3r",
    "T3_5r",
    "S_total_3r",
)


def _conditional(values: list[bool]) -> dict[str, int | float | None]:
    return {"n": len(values), "count": sum(values), "rate": mean(values) if values else None}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, object]:
    forwards = [row for row in rows if row.get("task") == "forward"]
    goals = [row for row in rows if row.get("task") in ("T4", "T5")]
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in forwards:
        families[row["family"]].append(row)
        regime = "spin" if row["family"] in ("side", "vertical") else "planar"
        families[regime].append(row)
    mechanisms: dict[str, object] = {}
    for family, members in sorted((dict(families) | {"all": forwards}).items()):
        if not members:
            continue
        scores = [row["scores"] for row in members]
        available_keys = [key for key in FORWARD_SUMMARY_KEYS if all(key in s for s in scores)]
        has_norm = all(
            "T1_per_ball" in s and "T2_cushion_set" in s and "T3_3r" in s for s in scores
        )
        true_positive_cases = [
            row for row in members if row["true_object_collision"] and row["valid"]
        ]
        mechanisms[family] = {
            "n": len(members),
            "valid": sum(row["valid"] for row in members),
            "mean": {key: mean(float(score[key]) for score in scores) for key in available_keys},
            "joint_event_and_all_coordinates": _conditional(
                [bool(score["T1"] and score["T2"] and score["T3_all_slots"]) for score in scores]
            ),
            "joint_normalized_event_and_3r": (
                _conditional(
                    [
                        bool(
                            score["T1_per_ball"] == 1.0
                            and score["T2_cushion_set"]
                            and score["T3_3r"] == 1.0
                        )
                        for score in scores
                    ]
                )
                if has_norm
                else None
            ),
            "all_coordinates_given_wrong_event": _conditional(
                [
                    bool(score["T3_all_slots"])
                    for score in scores
                    if not (score["T1"] and score["T2"])
                ]
            ),
            "stasis_given_true_collision_and_valid": _conditional(
                [bool(row["predicted_no_object_collision"]) for row in true_positive_cases]
            ),
            "invalid_given_true_collision": _conditional(
                [not row["valid"] for row in members if row["true_object_collision"]]
            ),
        }

    def attempts_curve(episodes: list[dict[str, Any]]) -> dict[str, object]:
        # v1 rows predate the recorded cap and always allowed six attempts.
        turns = max((row.get("max_attempts", 6) for row in episodes), default=6)
        return {
            str(turn): _conditional(
                [any(item["success"] for item in row["attempts"][:turn]) for row in episodes]
            )
            for turn in range(1, turns + 1)
        }

    def forecasts(task: str) -> dict[str, object]:
        items = [item for row in goals if row["task"] == task for item in row.get("forecasts", ())]
        scored = [item["event_agreement"] for item in items if item["event_agreement"] is not None]
        return {
            "coverage": _conditional([item["forecast_present"] for item in items]),
            "event_agreement": {"n": len(scored), "rate": mean(scored) if scored else None},
        }

    feedback = [row for row in goals if row["task"] == "T5"]
    single = [row for row in goals if row["task"] == "T4"]
    return {
        "forward": mechanisms,
        "T4": _conditional([row["success"] for row in single]),
        "T4_beats_no_shot": _conditional(
            [
                row["attempts"][0]["loss_radii"] is not None
                and row["attempts"][0]["loss_radii"] < row["baseline_radii"]
                for row in single
                if "baseline_radii" in row
            ]
        ),
        "T4_forecast": forecasts("T4"),
        "T5_forecast": forecasts("T5"),
        "T5_by_attempt": attempts_curve(feedback),
        "T5_by_regime_and_attempt": {
            "standard": attempts_curve([row for row in feedback if not row.get("ood")]),
            "unknown_friction": attempts_curve([row for row in feedback if row.get("ood")]),
        },
        "public_exact_answer": _conditional(
            [row["correct"] for row in rows if "idx" in row and "correct" in row]
        ),
        "public_episode_solved": _conditional(
            [row["solved"] for row in rows if "game" in row and "solved" in row]
        ),
    }


def _paired(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for metric in (
        *FORWARD_SUMMARY_KEYS,
        "T4",
        "T5",
        "public_exact_answer",
        "public_episode_solved",
    ):

        def by_id(rows: list[dict[str, Any]]) -> dict[str, tuple[str, float]]:
            if metric == "public_exact_answer":
                return {
                    str(row["idx"]): (str(row["idx"]), float(row["correct"]))
                    for row in rows
                    if "idx" in row and "correct" in row
                }
            if metric == "public_episode_solved":
                return {
                    str(row["game"]): (str(row["game"]), float(row["solved"]))
                    for row in rows
                    if "game" in row and "solved" in row
                }
            return {
                row["scene_id"]: (
                    row["group_id"],
                    float(row["success"] if metric in ("T4", "T5") else row["scores"][metric]),
                )
                for row in rows
                if row.get("task") == (metric if metric in ("T4", "T5") else "forward")
                and (metric in ("T4", "T5") or metric in row.get("scores", {}))
            }

        one, two = by_id(left), by_id(right)
        if not one or one.keys() != two.keys():
            continue
        if any(one[key][0] != two[key][0] for key in one):
            raise ValueError("Paired measurement group assignments disagree")
        result[metric] = summarize_paired(
            tuple(
                PairedMeasurement(pair_id=key, group_id=one[key][0], a=one[key][1], b=two[key][1])
                for key in sorted(one)
            ),
            seed=20261119,
            repetitions=10_000,
        ).model_dump(mode="json")
    return result


def analyze_runs(runs: dict[str, Path], output: Path) -> dict[str, object]:
    if not runs:
        raise ValueError("Select one or more completed runs")
    sources = {}
    comparison_keys: dict[str, tuple[object, ...]] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    summaries = {}
    for name, run in sorted(runs.items()):
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        run_hash = manifest.pop("run_hash")
        if fingerprint(manifest) != run_hash:
            raise ValueError(f"{name} has a modified protocol manifest")
        report = json.loads((run / "summary.json").read_text())
        if report["status"] != "complete" or report["run_hash"] != run_hash:
            raise ValueError(f"{name} has no matching completed summary")
        jobs = sorted((run / "jobs").glob("*.json"))
        extracted = [load_job(job, run_hash) for job in jobs]
        if len(extracted) != report.get("jobs", report.get("completed")) or any(
            row is None for row in extracted
        ):
            raise ValueError(f"{name} outcome ledger is incomplete")
        rows[name] = [row for row in extracted if row is not None]
        if (
            "data" in manifest
            or "valid_manifest_sha256" in manifest
            or ("split_sha256" in manifest)
        ):
            comparison_keys[name] = (
                "inverser-vlm",
                manifest.get("data")
                or manifest.get("valid_manifest_sha256")
                or manifest["split_sha256"],
            )
        else:
            comparison_keys[name] = (
                manifest.get("schema_version"),
                manifest.get("source_manifest_sha256"),
                manifest.get("asset_hashes"),
                manifest.get("dataset"),
            )
        summaries[name] = summarize_rows(rows[name])
        forward = [row for row in rows[name] if row.get("task") == "forward"]
        tokens = [
            reply.get("usage", {}).get("reasoning_tokens")
            for row in forward
            for reply in row.get("replies", ())
        ]
        sources[name] = {
            "manifest_sha256": file_sha256(manifest_path),
            "job_hashes_sha256": fingerprint({job.name: file_sha256(job) for job in jobs}),
            "reported_usage": report["usage"],
            "model_requests": report["counts"]["model_requests"],
            "forward_reasoning_tokens": (
                mean(tokens)
                if tokens
                and len(tokens) == len(forward)
                and all(value is not None for value in tokens)
                else None
            ),
            "evaluation_cohort": (
                fingerprint(sorted((row["scene_id"], row["group_id"]) for row in forward))
                if forward
                else None
            ),
        }
    names = sorted(rows)
    paired = {
        f"{names[i]}_minus_{names[j]}": _paired(rows[names[i]], rows[names[j]])
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if comparison_keys[names[i]] == comparison_keys[names[j]]
    }
    result: dict[str, object] = {
        "schema_version": "capo-analysis-v1",
        "sources": sources,
        "arms": summaries,
        "paired": {key: value for key, value in paired.items() if value},
        "new_model_requests": 0,
        "new_simulator_evaluations": 0,
    }
    create_artifact_dir(output, 5_000_000)
    result["figures"] = write_figures(output, summaries, sources)
    write_json(output / "analysis.json", result)
    with (output / "table.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("arm", "metric", "n", "rate"))
        for name in names:
            stats: Any = summaries[name]
            for metric in ("T4", "T4_beats_no_shot", "public_exact_answer"):
                writer.writerow((name, metric, stats[metric]["n"], stats[metric]["rate"]))
            solved = stats["public_episode_solved"]
            writer.writerow((name, "public_episode_solved", solved["n"], solved["rate"]))
            for task in ("T4", "T5"):
                for metric, values in stats[f"{task}_forecast"].items():
                    row = (name, f"{task}/forecast_{metric}", values["n"], values["rate"])
                    writer.writerow(row)
            for family, values in stats["forward"].items():
                for metric, rate in values["mean"].items():
                    writer.writerow((name, f"{family}/{metric}", values["n"], rate))
            for turn, values in stats["T5_by_attempt"].items():
                writer.writerow((name, f"T5/by_{turn}", values["n"], values["rate"]))
            for regime, turns in stats["T5_by_regime_and_attempt"].items():
                for turn, values in turns.items():
                    writer.writerow((name, f"T5/{regime}/by_{turn}", values["n"], values["rate"]))
    return result

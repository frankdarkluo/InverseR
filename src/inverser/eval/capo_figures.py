"""Deterministic vector figures from observed CAPO metrics; no placeholders for missing data."""

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

COLORS = ("#2165a8", "#a94b12", "#3c7857", "#733a98", "#505863")


def _text(x: float, y: float, value: str, *, size: int = 12, centered: bool = False) -> str:
    anchor = ' text-anchor="middle"' if centered else ""
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}"{anchor}>{escape(value)}</text>'


def _svg(path: Path, title: str, width: int, height: int, elements: list[str]) -> None:
    content = "\n".join(elements)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n<title>{escape(title)}</title>\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f'<g font-family="sans-serif" fill="#222733">\n{content}\n</g>\n</svg>\n'
    )


def write_figures(output: Path, summaries: dict[str, Any], sources: dict[str, Any]) -> list[str]:
    produced = []
    methods = [(name, data["forward"]) for name, data in summaries.items() if data["forward"]]
    if methods:
        metrics = ("T1", "T2", "T3", "S_total")
        height = 88 + 48 * len(methods)
        parts = [
            _text(20, 23, "Forward performance by physics regime", size=16),
            _text(20, 44, "Rows show observed evaluation scores (%); no missing cells imputed."),
        ]
        for column, entry in enumerate(
            f"{regime}/{metric}" for regime in ("planar", "spin") for metric in metrics
        ):
            parts.append(_text(233 + 72 * column, 70, entry, centered=True, size=10))
        for index, (arm, observed) in enumerate(methods):
            y: float = 81 + index * 48
            parts.append(_text(20, y + 22, arm[:29], size=11))
            for col, (regime, metric) in enumerate(
                (r, m) for r in ("planar", "spin") for m in metrics
            ):
                if regime not in observed:
                    continue
                score = observed[regime]["mean"][metric]
                shade = (int(242 - 193 * score), int(248 - 141 * score), int(254 - 69 * score))
                x: float = 198 + col * 72
                parts.append(
                    f'<rect x="{x}" y="{y}" width="70" height="32" '
                    f'fill="rgb({shade[0]},{shade[1]},{shade[2]})"/>'
                )
                ink = "white" if score > 0.72 else "#161a20"
                parts.append(
                    f'<text x="{x + 35}" y="{y + 21}" fill="{ink}" text-anchor="middle" '
                    f'font-size="12">{score * 100:.1f}</text>'
                )
        _svg(output / "regime_transfer.svg", "Forward regime matrix", 794, height, parts)
        produced.append("regime_transfer.svg")
        aliases = (
            ("Joint exact", "joint_event_and_all_coordinates"),
            ("All coords | wrong event", "all_coordinates_given_wrong_event"),
            ("Stasis | true collision and valid", "stasis_given_true_collision_and_valid"),
        )
        columns = [_text(20, 23, "Event–state alignment and stasis", size=16)]
        for col, (title, _) in enumerate(aliases):
            columns.append(_text(306 + col * 180, 51, title, size=11, centered=True))
        for i, (name, data) in enumerate(methods):
            y = 81 + i * 32
            columns.append(_text(20, y, name[:38], size=11))
            for col, (_, key) in enumerate(aliases):
                result = data["all"][key]
                rate, n = result["rate"], result["n"]
                label = f"{rate * 100:.1f}% · n={n}" if rate is not None else "not measured"
                columns.append(_text(306 + col * 180, y, label, centered=True))
        _svg(
            output / "causal_alignment.svg",
            "Observed event-state alignment",
            794,
            98 + 32 * len(methods),
            columns,
        )
        produced.append("causal_alignment.svg")

    feedback = [
        (name, data["T5_by_regime_and_attempt"]["unknown_friction"])
        for name, data in summaries.items()
        if data["T5_by_regime_and_attempt"]["unknown_friction"]["1"]["n"]
    ]
    if feedback:
        height = 280 + 19 * len(feedback)
        points = [_text(20, 24, "Unknown-friction adaptation (observed OOD episodes)", size=16)]
        for level in range(0, 101, 25):
            y = 240 - 1.8 * level
            points.extend(
                [f'<path d="M66 {y}H550" stroke="#dedfe3"/>', _text(27, y + 4, f"{level}%")]
            )
        turns = max(len(curve) for _, curve in feedback)  # the attempt cap: 6 in v1, 3 in v3
        for turn in range(1, turns + 1):
            points.append(_text(75 + 94 * (turn - 1), 258, str(turn), centered=True))
        points.append(_text(306, 278, "Attempts used (cumulative success)", centered=True))
        for i, (name, curve) in enumerate(feedback):
            color = COLORS[i % len(COLORS)]
            data = [
                (75 + 94 * (turn - 1), 240 - 180 * values["rate"])
                for turn, values in enumerate(curve.values(), 1)
            ]
            xy = " ".join(f"{x},{y:.2f}" for x, y in data)
            dash = ("0", "7 4", "2 4", "10 3 2 3")[i % 4]
            points.append(
                f'<polyline points="{xy}" fill="none" stroke="{color}" '
                f'stroke-width="2.1" stroke-dasharray="{dash}"/>'
            )
            points.extend(
                f'<circle cx="{x}" cy="{y:.2f}" r="2.5" fill="{color}"/>' for x, y in data
            )
            legend = 286 + 19 * i
            points.append(
                f'<path d="M56 {legend}H78" stroke="{color}" '
                f'stroke-width="2.1" stroke-dasharray="{dash}"/>'
            )
            points.append(_text(85, 290 + 19 * i, f"{name} · n={curve['6']['n']}"))
        _svg(
            output / "unknown_friction.svg",
            "Cumulative unknown-friction success",
            585,
            height + 12,
            points,
        )
        produced.append("unknown_friction.svg")

    # Keep only completed forward cohorts with observed, not inferred, reasoning usage.
    by_cohort: dict[object, list[tuple[str, float, float]]] = {}
    for name, stats in summaries.items():
        forward = stats["forward"].get("all")
        source = sources[name]
        tokens = source.get("forward_reasoning_tokens")
        cohort = source.get("evaluation_cohort")
        if forward and tokens is not None and cohort:
            by_cohort.setdefault(cohort, []).append((name, tokens, forward["mean"]["S_total"]))
    for number, (cohort, eligible) in enumerate(sorted(by_cohort.items())):
        if len(eligible) < 2:
            continue
        x_max = max(1.0, *(row[1] for row in eligible))
        marks = [
            _text(20, 27, "Forward accuracy–reasoning budget trade-off", size=16),
            _text(20, 49, f"Matched cohort {str(cohort)[:12]}; solid markers are non-dominated."),
        ]
        for score in range(0, 101, 25):
            y = 224 - 1.5 * score
            marks.extend(
                [f'<path d="M70 {y}H525" stroke="#dedfe3"/>', _text(26, y + 4, f"{score}%")]
            )
        marks.append(_text(74, 241, "0"))
        marks.append(_text(525, 241, f"{x_max:.0f}", centered=True))
        marks.append(
            _text(304, 262, "Mean reported reasoning tokens / forward request", centered=True)
        )
        for i, (name, tokens, score) in enumerate(eligible):
            x, y = 74 + 450 * tokens / x_max, 224 - 150 * score
            color = COLORS[i % len(COLORS)]
            dominated = any(
                tx <= tokens and other >= score and (tx < tokens or other > score)
                for _, tx, other in eligible
            )
            fill = "white" if dominated else color
            marks.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.3" '
                f'fill="{fill}" stroke="{color}" stroke-width="2"/>'
            )
            marks.append(
                _text(
                    75,
                    283 + 17 * i,
                    f"{i + 1}. {name}: {tokens:.1f} tokens; {score * 100:.2f}% S_total",
                    size=11,
                )
            )
            marks.append(_text(x + (-12 if x > 510 else 7), y - 6, str(i + 1), size=10))
        filename = f"reasoning_pareto_{number}.svg"
        _svg(
            output / filename,
            "Matched forward reasoning trade-off",
            574,
            294 + 17 * len(eligible),
            marks,
        )
        produced.append(filename)
    return produced

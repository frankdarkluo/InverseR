import json
import subprocess
import tempfile
from datetime import UTC, datetime
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw
from pydantic import Field

from inverser.runtime import check_storage, file_sha256, write_json
from inverser.schemas import Record


class SourceAsset(Record):
    asset_id: str = Field(pattern=r"^[a-z0-9_]+$")
    url: str
    rights_page: str
    expected_bytes: int = Field(gt=0, le=100_000_000)


class Source(Record):
    source_id: str = Field(pattern=r"^[a-z0-9_]+$")
    url: str
    game: Literal["snooker"]
    license: str
    license_evidence: str
    credit: str
    acquire: Literal["allowed", "unverified", "denied"]
    process: Literal["allowed", "unverified", "denied"]
    redistribute: Literal["allowed", "unverified", "denied"]
    access: Literal["ready", "blocked"]
    technical_access: Literal["reachable", "blocked", "unverified"] = "unverified"
    local_download_requested: bool = False
    disposition: str
    assets: tuple[SourceAsset, ...]


class VideoInfo(Record):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_s: float = Field(gt=0)
    fps: float = Field(gt=0)
    codec: str


def load_sources(path: Path) -> tuple[Source, ...]:
    raw = json.loads(path.read_text())
    if raw.get("schema_version") != "source-registry-v1":
        raise ValueError("Unsupported source-registry version")
    return tuple(Source.model_validate(source) for source in raw["sources"])


def require_acquisition(source: Source, asset: SourceAsset) -> None:
    if source.acquire != "allowed" or source.process != "allowed" or source.access != "ready":
        raise PermissionError(f"{source.source_id}: acquisition is not qualified")
    if asset not in source.assets:
        raise PermissionError("Asset is not in the verified source manifest")
    parsed = urlparse(asset.url)
    if parsed.scheme != "https" or parsed.hostname != "upload.wikimedia.org":
        raise PermissionError("Pilot downloader permits only the verified Commons media host")


def probe_video(path: Path) -> VideoInfo:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    data = json.loads(completed.stdout)
    videos = [stream for stream in data["streams"] if stream["codec_type"] == "video"]
    if len(videos) != 1:
        raise ValueError("Expected exactly one video stream")
    video = videos[0]
    duration = float(data["format"]["duration"])
    fps = float(Fraction(video["avg_frame_rate"]))
    if duration <= 0 or fps <= 0:
        raise ValueError("Video duration and frame rate must be positive")
    return VideoInfo(
        width=int(video["width"]),
        height=int(video["height"]),
        duration_s=duration,
        fps=fps,
        codec=video["codec_name"],
    )


def acquire_asset(source: Source, asset: SourceAsset, output: Path) -> dict[str, object]:
    require_acquisition(source, asset)
    check_storage(asset.expected_bytes * 2)
    output.mkdir(parents=True, exist_ok=True)
    media = output / f"{asset.asset_id}.ogv"
    receipt_path = output / f"{asset.asset_id}.json"
    if media.exists():
        if not receipt_path.exists():
            raise ValueError("Existing media has no acquisition receipt; refusing to overwrite")
        receipt = json.loads(receipt_path.read_text())
        if receipt["url"] != asset.url or receipt["sha256"] != file_sha256(media):
            raise ValueError("Cached media does not match its acquisition receipt")
        return receipt
    temporary: Path | None = None
    request = Request(asset.url, headers={"User-Agent": "InverseR-research-source-pilot/0.1"})
    try:
        with urlopen(request, timeout=30) as response:
            final_url = urlparse(response.geturl())
            if final_url.scheme != "https" or final_url.hostname != "upload.wikimedia.org":
                raise PermissionError("Unexpected download redirect")
            content_type = response.headers.get_content_type()
            if content_type not in ("application/ogg", "video/ogg"):
                raise ValueError(f"Expected OGV media, received {content_type}")
            with tempfile.NamedTemporaryFile(
                "wb", dir=output, prefix=f".{asset.asset_id}.", delete=False
            ) as stream:
                temporary = Path(stream.name)
                received = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    received += len(chunk)
                    if received > asset.expected_bytes:
                        raise ValueError("Asset exceeds its verified byte-size bound")
                    stream.write(chunk)
        if received != asset.expected_bytes:
            raise ValueError("Incomplete download: byte size differs from verified manifest")
        video = probe_video(temporary)
        receipt = {
            "source_id": source.source_id,
            "asset_id": asset.asset_id,
            "url": asset.url,
            "rights_page": asset.rights_page,
            "license": source.license,
            "credit": source.credit,
            "bytes": received,
            "sha256": file_sha256(temporary),
            "retrieved_utc": datetime.now(UTC).isoformat(),
            "video": video.model_dump(mode="json"),
            "qa_state": "content_unqualified",
            "accepted_shot_tasks": 0,
        }
        temporary.replace(media)
        write_json(receipt_path, receipt)
        return receipt
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def video_contact_sheet(media: Path, output: Path) -> None:
    duration = probe_video(media).duration_s
    sheet = Image.new("RGB", (640, 4 * 385), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for index, fraction in enumerate((0.05, 0.35, 0.65, 0.95)):
        timestamp = duration * fraction
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-threads",
                "1",
                "-filter_threads",
                "1",
                "-ss",
                str(timestamp),
                "-i",
                str(media),
                "-frames:v",
                "1",
                "-vf",
                "scale=640:-1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        with Image.open(BytesIO(result.stdout)) as image:
            sheet.paste(image.convert("RGB"), (0, index * 385 + 25))
        draw.text((8, index * 385 + 5), f"{timestamp:.2f}s", fill="#202020")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def acquire_source_pilot(registry: Path, output: Path) -> dict[str, object]:
    receipts = []
    blocked = []
    for source in load_sources(registry):
        if source.access != "ready" or source.acquire != "allowed" or source.process != "allowed":
            blocked.append({"source_id": source.source_id, "reason": source.disposition})
            continue
        for asset in source.assets:
            receipts.append(acquire_asset(source, asset, output))
            video_contact_sheet(
                output / f"{asset.asset_id}.ogv", output / f"{asset.asset_id}_contact.png"
            )
    report = {
        "downloaded_assets": receipts,
        "blocked_sources": blocked,
        "accepted_shot_tasks": 0,
        "quality_state": "media acquired; complete shot/trajectory qualification still required",
    }
    write_json(output / "acquisition.json", report)
    return report

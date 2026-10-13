from dataclasses import dataclass

import cv2
import numpy as np

from inverser.vision import ImageArray


@dataclass(frozen=True)
class TableLocation:
    points: tuple[tuple[float, float], ...]
    frame_area_fraction: float
    rectangularity: float
    long_axis_horizontal: bool


def locate_table(frame: ImageArray) -> TableLocation | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.asarray([30, 40, 35]), np.asarray([100, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = frame.shape[0] * frame.shape[1] * 0.12
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < area_min:
            continue
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0 or area / hull_area < 0.70:
            continue
        perimeter = cv2.arcLength(hull, True)
        approximation = None
        for epsilon in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04):
            candidate = cv2.approxPolyDP(hull, epsilon * perimeter, True)
            if len(candidate) == 4:
                approximation = candidate[:, 0, :].astype(np.float32)
                break
        if approximation is None or not cv2.isContourConvex(approximation):
            continue
        if (
            np.any(approximation[:, 0] < 3)
            or np.any(approximation[:, 1] < 3)
            or np.any(approximation[:, 0] >= frame.shape[1] - 3)
            or np.any(approximation[:, 1] >= frame.shape[0] - 3)
        ):
            continue
        sums = approximation.sum(axis=1)
        differences = approximation[:, 0] - approximation[:, 1]
        tl = approximation[int(np.argmin(sums))]
        br = approximation[int(np.argmax(sums))]
        tr = approximation[int(np.argmax(differences))]
        bl = approximation[int(np.argmin(differences))]
        if len({tuple(point) for point in (tl, tr, br, bl)}) != 4:
            continue
        horizontal = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        vertical = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        far_edge = float(np.linalg.norm(tr - tl))
        near_edge = float(np.linalg.norm(br - bl))
        perspective_trapezoid = abs(far_edge - near_edge) > max(far_edge, near_edge) * 0.2
        long_horizontal = bool(horizontal > vertical * 1.2 and not perspective_trapezoid)
        ordered = (tl, bl, br, tr) if long_horizontal else (tl, tr, br, bl)
        return TableLocation(
            points=tuple((float(point[0]), float(point[1])) for point in ordered),
            frame_area_fraction=float(area / (frame.shape[0] * frame.shape[1])),
            rectangularity=float(area / hull_area),
            long_axis_horizontal=long_horizontal,
        )
    return None

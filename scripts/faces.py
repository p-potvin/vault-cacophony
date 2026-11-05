#!/usr/bin/env python3
"""The face pass: SCRFD detection, GlintR100 clustering, gallery matching, and exemplar picking.

Every video pass in this repo writes its own track into the media's sidecar `tags`
(`tagstore.py`) and leaves the others alone. This module is the visual
counterpart to `voices.py`:

1. **Detection & Embedding (`antelopev2`)**
   Runs ColONEL-KFC's InsightFace `antelopev2` contract (`SCRFD-10G` detector,
   `GlintR100` 512-d L2-normalized embedder, `2D-106` landmarks) over sampled
   video frames.
2. **Per-Video Clustering & Gallery Identification**
   Clusters face detections across the video by cosine similarity so a recurring
   person gets one coherent track (`FACE_0`, `FACE_1`, ...), then matches each
   cluster (and individual observations) against ColONEL-KFC's `gallery.db`
   (reaching it through `vault-raggae` `/v1/sql` when the gateway owns the
   store, or read-only SQLite when local).
3. **2D-106 Yaw Estimation & Exemplar Scoring**
   Scores each detection for downstream avatar/mesh reconstruction (LAM,
   MICA/FLAME, trellis.cpp):
   - embedding cosine similarity to the identity's video centroid,
   - frontal head pose (`abs(yaw_deg) <= max_yaw_deg` from 2D-106 / 5-point landmarks),
   - bounding-box area (`bbox_area`),
   - detection probability, with temporal separation so exemplars span the clip.
4. **Tag Store Track (`face`)**
   Writes collapsed time spans `[{start, end, value, similarity, bbox, yaw_deg}]`
   and per-identity exemplar summaries into the media's sidecar `tags`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tagstore  # noqa: E402

ANTELOPEV2_CONTRACT = "insightface-antelopev2:glintr100+scrfd10g+2d106+genderage"
EMB_DIM = 512
DEFAULT_MATCH_THRESHOLD = 0.45
DEFAULT_CLUSTER_THRESHOLD = 0.45
DEFAULT_MAX_YAW_DEG = 25.0
DEFAULT_MIN_BOX_PX = 48.0
DEFAULT_MIN_PROB = 0.60


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12 or not np.all(np.isfinite(arr)):
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32)


@dataclass
class FaceObservation:
    """One detected face on one sampled video frame."""

    timestamp: float
    frame_path: str
    embedding: np.ndarray
    bbox: List[float]                  # [x1, y1, x2, y2] in pixels
    prob: float = 1.0
    feature_norm: float = 20.0
    landmarks_5pts: Optional[np.ndarray] = None
    landmarks_106: Optional[np.ndarray] = None
    chunk_idx: Optional[int] = None
    identity: Optional[str] = None
    similarity: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    exemplar_score: float = 0.0

    @property
    def box_width(self) -> float:
        return max(0.0, float(self.bbox[2]) - float(self.bbox[0])) if len(self.bbox) >= 4 else 0.0

    @property
    def box_height(self) -> float:
        return max(0.0, float(self.bbox[3]) - float(self.bbox[1])) if len(self.bbox) >= 4 else 0.0

    @property
    def box_area(self) -> float:
        return self.box_width * self.box_height


# ----------------------------------------------------------------------------
# Landmark pose estimation (2D-106 and 5-point ArcFace geometry)


def estimate_pose_degrees(
    landmarks_5pts: Optional[np.ndarray] = None,
    landmarks_106: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Estimate head yaw and roll in degrees from 2D-106 or 5-point landmarks.

    0 deg yaw is dead-frontal; positive/negative values indicate left/right turn.
    """
    yaw_deg = 0.0
    roll_deg = 0.0

    if landmarks_106 is not None:
        pts106 = np.asarray(landmarks_106, dtype=np.float32)
        if pts106.ndim == 2 and pts106.shape[0] >= 106 and pts106.shape[1] >= 2:
            # InsightFace 2D-106 layout:
            #   0..32: jaw contour (1 = left cheek edge, 31 = right cheek edge, 16 = chin)
            #   33..42 / 87..96: left / right eye contours
            #   86: nose tip
            left_cheek = pts106[1, :2]
            right_cheek = pts106[31, :2]
            nose_tip = pts106[86, :2]
            left_eye = np.mean(pts106[33:43, :2], axis=0)
            right_eye = np.mean(pts106[87:97, :2], axis=0)

            dx_eye = float(right_eye[0] - left_eye[0])
            dy_eye = float(right_eye[1] - left_eye[1])
            if abs(dx_eye) > 1e-3 or abs(dy_eye) > 1e-3:
                roll_deg = math.degrees(math.atan2(dy_eye, dx_eye))

            # Project nose offset onto the inter-cheek axis to stay roll-invariant
            cheek_vec = right_cheek - left_cheek
            cheek_span = float(np.linalg.norm(cheek_vec))
            if cheek_span > 1e-3:
                unit_axis = cheek_vec / cheek_span
                mid_cheek = 0.5 * (left_cheek + right_cheek)
                rel_offset = float(np.dot(nose_tip - mid_cheek, unit_axis)) / (0.5 * cheek_span)
                rel_offset = max(-1.0, min(1.0, rel_offset))
                yaw_deg = math.degrees(math.asin(rel_offset))
                return {"yaw_deg": round(yaw_deg, 2), "roll_deg": round(roll_deg, 2)}

    if landmarks_5pts is not None:
        pts5 = np.asarray(landmarks_5pts, dtype=np.float32)
        if pts5.ndim == 2 and pts5.shape[0] >= 5 and pts5.shape[1] >= 2:
            left_eye, right_eye, nose = pts5[0, :2], pts5[1, :2], pts5[2, :2]
            eye_vec = right_eye - left_eye
            eye_span = float(np.linalg.norm(eye_vec))
            if eye_span > 1e-3:
                roll_deg = math.degrees(math.atan2(float(eye_vec[1]), float(eye_vec[0])))
                unit_axis = eye_vec / eye_span
                mid_eye = 0.5 * (left_eye + right_eye)
                # Nose horizontal offset relative to half inter-ocular distance
                rel_offset = float(np.dot(nose - mid_eye, unit_axis)) / (0.65 * eye_span)
                rel_offset = max(-1.0, min(1.0, rel_offset))
                yaw_deg = math.degrees(math.asin(rel_offset))

    return {"yaw_deg": round(yaw_deg, 2), "roll_deg": round(roll_deg, 2)}


# ----------------------------------------------------------------------------
# Gallery bridge (ColONEL-KFC gallery.db via vault-raggae or read-only SQLite)


def default_gallery_paths() -> List[Path]:
    """Candidate ColONEL-KFC gallery.db locations in priority order."""
    candidates: List[Path] = []
    env_gal = os.environ.get("VW_FACE_GALLERY")
    if env_gal:
        p = Path(env_gal)
        candidates.append(p if p.suffix.lower() == ".db" else p / "gallery.db")
    for raw in (
        r"G:\Gallery\gallery.db",
        r"G:\Galleries\Celebrities\gallery.db",
        r"F:\amd\gallery\gallery.db",
        r"D:\miniville\gallery\gallery.db",
    ):
        candidates.append(Path(raw))
    out: List[Path] = []
    seen = set()
    for c in candidates:
        norm = os.path.normcase(os.path.abspath(str(c)))
        if norm not in seen:
            seen.add(norm)
            out.append(c)
    return out


class FaceGallery:
    """In-memory multi-vector matcher loaded from a ColONEL-KFC `gallery.db`."""

    def __init__(
        self,
        names: Sequence[str],
        embeddings: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None,
        min_similarity: float = DEFAULT_MATCH_THRESHOLD,
        source_path: str = "",
    ):
        self.source_path = source_path
        self.min_similarity = float(min_similarity)
        self.names = list(names)
        if len(self.names) > 0:
            mat = np.asarray(embeddings, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms > 1e-12, norms, 1.0)
            self.matrix = (mat / norms).astype(np.float32)
        else:
            self.matrix = np.zeros((0, EMB_DIM), dtype=np.float32)

        self.unique_names = sorted(set(self.names))
        self.name_to_idx = {n: i for i, n in enumerate(self.unique_names)}
        self.crop_model_indices = np.asarray(
            [self.name_to_idx[n] for n in self.names], dtype=np.int32
        ) if self.names else np.zeros((0,), dtype=np.int32)
        raw_thresh = thresholds or {}
        self.thresholds = {
            n: max(self.min_similarity, float(raw_thresh.get(n, self.min_similarity)))
            for n in self.unique_names
        }

    @property
    def empty(self) -> bool:
        return self.matrix.shape[0] == 0

    @classmethod
    def load(
        cls,
        gallery_path: Optional[str | os.PathLike] = None,
        min_similarity: float = DEFAULT_MATCH_THRESHOLD,
    ) -> "FaceGallery":
        """Load active embeddings from a `gallery.db` file or directory (or return an empty gallery)."""
        db_path: Optional[Path] = None
        if gallery_path:
            p = Path(gallery_path)
            db_path = p if p.suffix.lower() == ".db" else p / "gallery.db"
        else:
            for cand in default_gallery_paths():
                if cand.is_file():
                    db_path = cand
                    break

        if db_path is None or not db_path.is_file():
            return cls([], np.zeros((0, EMB_DIM), dtype=np.float32), min_similarity=min_similarity)

        # Prefer ColONEL-KFC's GalleryDB (which routes through vault-raggae raggae_sql when owned)
        kfc_root = Path(os.environ.get(
            "COLONEL_KFC_PATH",
            os.path.join(os.path.expanduser("~"), "Desktop", "Github Repos", "ColONEL-KFC"),
        ))
        if kfc_root.is_dir() and str(kfc_root) not in sys.path:
            sys.path.insert(0, str(kfc_root))
        try:
            from face_organizer.db import GalleryDB  # type: ignore
            gdb = GalleryDB(str(db_path))
            names, matrix = gdb.get_all_active_embeddings()
            thresholds = {
                row["name"]: float(row.get("threshold") or min_similarity)
                for row in gdb.get_all_identities()
            }
            return cls(names, matrix, thresholds=thresholds, min_similarity=min_similarity, source_path=str(db_path))
        except Exception:
            pass

        # Direct read-only SQLite fallback
        names_list: List[str] = []
        vecs: List[np.ndarray] = []
        thresholds_map: Dict[str, float] = {}
        uri = db_path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=15.0)
        try:
            for row in conn.execute("SELECT name, threshold FROM identities"):
                thresholds_map[str(row[0])] = float(row[1] or min_similarity)
            sql = """
                SELECT fc.model_name, fc.embedding
                FROM face_crops fc
                LEFT JOIN identities i ON fc.identity_id = i.id
                WHERE i.status IS NULL OR i.status != 'quarantined'
            """
            for model_name, blob in conn.execute(sql):
                if not blob:
                    continue
                v = np.frombuffer(blob, dtype=np.float32)
                if v.shape == (EMB_DIM,) and np.all(np.isfinite(v)):
                    names_list.append(str(model_name))
                    vecs.append(_unit(v))
        finally:
            conn.close()

        mat = np.vstack(vecs) if vecs else np.zeros((0, EMB_DIM), dtype=np.float32)
        return cls(names_list, mat, thresholds=thresholds_map, min_similarity=min_similarity, source_path=str(db_path))

    def match(self, embedding: np.ndarray, margin: float = 0.02) -> Tuple[Optional[str], float]:
        """Return `(identity_name, cosine_score)` or `(None, best_score)` if below threshold."""
        if self.empty:
            return None, 0.0
        vec = _unit(embedding)
        if vec.shape != (EMB_DIM,) or float(np.linalg.norm(vec)) <= 1e-6:
            return None, 0.0
        sims = np.dot(self.matrix, vec)
        best_per_id = np.full(len(self.unique_names), -np.inf, dtype=np.float32)
        np.maximum.at(best_per_id, self.crop_model_indices, sims)
        order = np.argsort(best_per_id)[::-1]
        best_idx = int(order[0])
        best_name = self.unique_names[best_idx]
        best_score = float(best_per_id[best_idx])
        thresh = self.thresholds.get(best_name, self.min_similarity)
        if best_score < thresh:
            return None, round(best_score, 4)
        if len(order) > 1:
            runner_up = float(best_per_id[int(order[1])])
            if best_score - runner_up < margin and best_score < thresh + 0.08:
                return None, round(best_score, 4)
        return best_name, round(best_score, 4)


# ----------------------------------------------------------------------------
# Clustering & Exemplar Scoring


def cluster_observations(
    observations: Sequence[FaceObservation],
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    gallery: Optional[FaceGallery] = None,
) -> Tuple[List[FaceObservation], Dict[str, Dict[str, Any]]]:
    """Group face detections into identity tracks, match against `gallery`, and score exemplars.

    Returns `(annotated_observations, identities_summary)`.
    """
    if not observations:
        return [], {}

    # Filter out invalid embeddings and annotate pose
    valid: List[FaceObservation] = []
    for obs in observations:
        vec = _unit(obs.embedding)
        if vec.shape != (EMB_DIM,) or float(np.linalg.norm(vec)) <= 1e-6:
            continue
        pose = estimate_pose_degrees(obs.landmarks_5pts, obs.landmarks_106)
        obs.embedding = vec
        obs.yaw_deg = pose["yaw_deg"]
        obs.roll_deg = pose["roll_deg"]
        valid.append(obs)

    if not valid:
        return [], {}

    # Greedy quality-first clustering (highest prob * area seeds each cluster)
    ordered = sorted(
        valid,
        key=lambda o: float(o.prob) * math.sqrt(max(1.0, o.box_area)) * max(0.2, 1.0 - abs(o.yaw_deg) / 90.0),
        reverse=True,
    )
    clusters: List[List[FaceObservation]] = []
    centroids: List[np.ndarray] = []

    for obs in ordered:
        best_c = -1
        best_sim = -1.0
        for ci, c_vec in enumerate(centroids):
            sim = float(np.dot(c_vec, obs.embedding))
            if sim >= cluster_threshold and sim > best_sim:
                best_sim = sim
                best_c = ci
        if best_c >= 0:
            clusters[best_c].append(obs)
            centroids[best_c] = _unit(np.mean([x.embedding for x in clusters[best_c]], axis=0))
        else:
            clusters.append([obs])
            centroids.append(obs.embedding.copy())

    # Sort clusters by frequency (most screen time first)
    clusters_with_centroids = sorted(
        zip(clusters, centroids),
        key=lambda pair: (-len(pair[0]), min(o.timestamp for o in pair[0])),
    )

    # Match each cluster against gallery (and merge two clusters if both match the same gallery identity)
    by_identity: Dict[str, List[FaceObservation]] = {}
    identity_match_meta: Dict[str, Tuple[str, float]] = {}
    unknown_counter = 0

    for cluster, centroid in clusters_with_centroids:
        matched_name: Optional[str] = None
        matched_score = 0.0
        if gallery is not None and not gallery.empty:
            matched_name, matched_score = gallery.match(centroid)
            if not matched_name:
                # Check if the highest-quality frontal exemplar in the cluster matches
                best_obs = cluster[0]
                matched_name, matched_score = gallery.match(best_obs.embedding)

        if matched_name:
            label = matched_name
            prev = identity_match_meta.get(label)
            if prev is None or matched_score > prev[1]:
                identity_match_meta[label] = ("gallery", matched_score)
        else:
            label = f"FACE_{unknown_counter}"
            unknown_counter += 1
            identity_match_meta[label] = ("cluster", 0.0)

        by_identity.setdefault(label, []).extend(cluster)

    # Compute final per-identity centroids, exemplar scores, and summaries
    identities_summary: Dict[str, Dict[str, Any]] = {}
    all_annotated: List[FaceObservation] = []

    for label, group in by_identity.items():
        group.sort(key=lambda o: o.timestamp)
        centroid = _unit(np.mean([o.embedding for o in group], axis=0))
        max_area = max((o.box_area for o in group), default=1.0) or 1.0
        source_kind, gallery_sim = identity_match_meta.get(label, ("cluster", 0.0))

        for obs in group:
            c_sim = float(np.dot(centroid, obs.embedding))
            obs.identity = label
            obs.similarity = round(gallery_sim if source_kind == "gallery" else c_sim, 4)
            obs.exemplar_score = score_exemplar(obs, centroid_sim=c_sim, max_box_area=max_area)
            all_annotated.append(obs)

        exemplars = pick_exemplars(group, centroid=centroid, top_k=5)
        timestamps = [o.timestamp for o in group]
        identities_summary[label] = {
            "identity": label,
            "source": source_kind,
            "gallery_similarity": round(gallery_sim, 4) if source_kind == "gallery" else None,
            "observations": len(group),
            "first_s": round(min(timestamps), 3),
            "last_s": round(max(timestamps), 3),
            "exemplars": [
                {
                    "timestamp": round(ex.timestamp, 3),
                    "frame_path": os.path.basename(ex.frame_path) if ex.frame_path else "",
                    "score": round(ex.exemplar_score, 4),
                    "yaw_deg": round(ex.yaw_deg, 2),
                    "bbox": [round(float(x), 1) for x in ex.bbox],
                    "prob": round(float(ex.prob), 4),
                }
                for ex in exemplars
            ],
        }

    all_annotated.sort(key=lambda o: (o.timestamp, o.identity or ""))
    return all_annotated, identities_summary


def score_exemplar(
    obs: FaceObservation,
    centroid_sim: float,
    max_box_area: float = 10000.0,
    max_yaw_deg: float = DEFAULT_MAX_YAW_DEG,
) -> float:
    """Score a detection as a reconstruction exemplar (centroid cosine + frontal yaw + box size)."""
    yaw_abs = abs(float(obs.yaw_deg))
    if yaw_abs <= max_yaw_deg:
        frontal_factor = 1.0 - 0.35 * (yaw_abs / max(1.0, max_yaw_deg))
    else:
        # Heavy penalty once yaw exceeds threshold
        frontal_factor = max(0.05, 0.35 * (1.0 - (yaw_abs - max_yaw_deg) / 65.0))

    roll_penalty = max(0.2, 1.0 - abs(float(obs.roll_deg)) / 60.0)
    size_ratio = math.sqrt(max(0.0, obs.box_area) / max(1.0, max_box_area))
    size_factor = min(1.0, max(0.15, size_ratio))
    c_factor = max(0.0, min(1.0, float(centroid_sim)))
    prob_factor = max(0.1, min(1.0, float(obs.prob)))

    score = (0.45 * c_factor + 0.30 * frontal_factor + 0.25 * size_factor) * roll_penalty * prob_factor
    if yaw_abs > max_yaw_deg:
        score *= 0.4
    return round(float(score), 4)


def pick_exemplars(
    observations: Sequence[FaceObservation],
    centroid: Optional[np.ndarray] = None,
    top_k: int = 5,
    max_yaw_deg: float = DEFAULT_MAX_YAW_DEG,
    min_time_separation_s: float = 1.0,
) -> List[FaceObservation]:
    """Select up to `top_k` best frontal, high-centrality, temporally distinct exemplars."""
    if not observations:
        return []
    if centroid is None:
        centroid = _unit(np.mean([o.embedding for o in observations], axis=0))
    max_area = max((o.box_area for o in observations), default=1.0) or 1.0

    for obs in observations:
        c_sim = float(np.dot(centroid, _unit(obs.embedding)))
        obs.exemplar_score = score_exemplar(obs, centroid_sim=c_sim, max_box_area=max_area, max_yaw_deg=max_yaw_deg)

    # Prefer frontal detections (abs(yaw) <= max_yaw_deg) if available
    frontal = [o for o in observations if abs(o.yaw_deg) <= max_yaw_deg]
    pool = frontal if frontal else list(observations)
    ranked = sorted(pool, key=lambda o: o.exemplar_score, reverse=True)

    picked: List[FaceObservation] = []
    for cand in ranked:
        if len(picked) >= top_k:
            break
        if all(abs(cand.timestamp - p.timestamp) >= min_time_separation_s for p in picked):
            picked.append(cand)

    if len(picked) < min(top_k, len(ranked)):
        for cand in ranked:
            if len(picked) >= top_k:
                break
            if cand not in picked:
                picked.append(cand)

    return picked


# ----------------------------------------------------------------------------
# Spans & Tag Store integration


def observations_to_spans(
    observations: Sequence[FaceObservation],
    window_half_s: float = 0.75,
    merge_gap_s: float = 3.5,
    duration: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Turn frame-level face observations into merged time spans for `tagstore.put_track('face', ...)`."""
    if not observations:
        return []

    by_id: Dict[str, List[FaceObservation]] = {}
    for obs in observations:
        if obs.identity:
            by_id.setdefault(obs.identity, []).append(obs)

    spans: List[Dict[str, Any]] = []
    for ident, group in by_id.items():
        group.sort(key=lambda o: o.timestamp)
        cur_start = max(0.0, group[0].timestamp - window_half_s)
        cur_end = group[0].timestamp + window_half_s
        cur_obs = [group[0]]

        def _flush(s: float, e: float, items: List[FaceObservation]) -> Dict[str, Any]:
            if duration is not None and duration > 0:
                e = min(float(duration), e)
            best = max(items, key=lambda x: x.exemplar_score)
            return {
                "start": round(s, 3),
                "end": round(max(s, e), 3),
                "value": ident,
                "similarity": round(max(x.similarity for x in items), 4),
                "hits": len(items),
                "best_timestamp": round(best.timestamp, 3),
                "yaw_deg": round(best.yaw_deg, 2),
                "bbox": [round(float(v), 1) for v in best.bbox],
            }

        for obs in group[1:]:
            s = max(0.0, obs.timestamp - window_half_s)
            e = obs.timestamp + window_half_s
            if s - cur_end <= merge_gap_s:
                cur_end = max(cur_end, e)
                cur_obs.append(obs)
            else:
                spans.append(_flush(cur_start, cur_end, cur_obs))
                cur_start, cur_end, cur_obs = s, e, [obs]
        spans.append(_flush(cur_start, cur_end, cur_obs))

    spans.sort(key=lambda sp: (sp["start"], sp["end"], sp["value"]))
    return spans


def write_face_track(
    video: str,
    observations: Sequence[FaceObservation],
    identities: Dict[str, Dict[str, Any]],
    duration: Optional[float] = None,
    model: str = ANTELOPEV2_CONTRACT,
    gallery_path: str = "",
) -> str:
    """Write the `face` track into the video's sidecar `tags` without disturbing other tracks or keys."""
    store = tagstore.load(video)
    tagstore.set_media(store, video, duration)
    spans = observations_to_spans(observations, duration=duration)
    extra: Dict[str, Any] = {}
    if identities:
        extra["identities"] = list(identities.values())
    if gallery_path:
        extra["gallery"] = os.path.basename(gallery_path)
    tagstore.put_track(store, "face", spans, "face", model=model, **extra)
    return tagstore.save(store)


# ----------------------------------------------------------------------------
# Detector loader (ColONEL-KFC FaceEngine or InsightFace antelopev2)


class AntelopeFaceDetector:
    """Callable detector `(frame_path, timestamp, chunk_idx) -> List[FaceObservation]`."""

    def __init__(
        self,
        device: str = "cuda",
        min_prob: float = DEFAULT_MIN_PROB,
        min_box_px: float = DEFAULT_MIN_BOX_PX,
    ):
        self.device = device
        self.min_prob = float(min_prob)
        self.min_box_px = float(min_box_px)
        self._engine = None

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        kfc_root = Path(os.environ.get(
            "COLONEL_KFC_PATH",
            os.path.join(os.path.expanduser("~"), "Desktop", "Github Repos", "ColONEL-KFC"),
        ))
        if kfc_root.is_dir() and str(kfc_root) not in sys.path:
            sys.path.insert(0, str(kfc_root))
        from face_organizer.face_engine import FaceEngine  # type: ignore
        self._engine = FaceEngine(
            device=self.device,
            analysis_modules=("detection", "recognition", "landmark_2d_106"),
        )
        return self._engine

    def detect_frame(
        self,
        frame_path: str,
        timestamp: float,
        chunk_idx: Optional[int] = None,
    ) -> List[FaceObservation]:
        engine = self._ensure_engine()
        results = engine.extract_all_face_details(frame_path, auto_rotate=False)
        out: List[FaceObservation] = []
        for r in results:
            bbox = [float(v) for v in r.bbox]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if float(r.prob) < self.min_prob or min(w, h) < self.min_box_px:
                continue
            out.append(
                FaceObservation(
                    timestamp=float(timestamp),
                    frame_path=frame_path,
                    embedding=np.asarray(r.embedding, dtype=np.float32),
                    bbox=bbox,
                    prob=float(r.prob),
                    feature_norm=float(r.feature_norm),
                    landmarks_5pts=r.landmarks_5pts,
                    landmarks_106=r.landmarks_106,
                    chunk_idx=chunk_idx,
                )
            )
        return out

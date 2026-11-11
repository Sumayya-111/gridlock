"""
api/main.py — Gridlock FastAPI Backend
  Wraps the existing src/ detection pipeline behind REST + WebSocket endpoints.
  Models are lazy-loaded as singletons (expensive YOLO / PaddleOCR weights
  only load on first request, not at import time).

  Violations on the video WebSocket: NO_HELMET, TRIPLE_RIDING, SIGNAL_JUMP, WRONG_WAY.

Usage:
  uvicorn api.main:app --reload --port 8000 --ws-max-size 1073741824   (1 GB; default 16 MB breaks video upload)

Helmet debugging:
  HELMET_DEBUG=true          print per-rider readings, vote counts, and suppressed alerts
  HELMET_INCLUDE_BICYCLE=true  also check pedal cyclists (default: motorcycles only)
  LOOK_DEDUPE=true           enable same-looking-rider filter (only for MOVING cameras)

Signal-jump config (coordinates are in the RESIZED frame, max width 1280):
  Option A (env, no frontend change needed):
      SIGNAL_ROI="x1,y1,x2,y2"            box around the traffic light (";" for several lamps)
      STOP_LINE="x1,y1,x2,y2"             two points of the stop line  (";" for several lines)
      SIGNAL_DIR=any|forward|reverse      which crossing direction counts (default: any)
      SIGNAL_DEBUG=true                   print signal state + vehicles near the line
  Option B (per video, WebSocket query params — override the env values):
      /ws/detect/video?signal_roi=x1,y1,x2,y2&stop_line=x1,y1,x2,y2&stop_dir=any

  Tips:
    - Draw each stop line LEFT -> RIGHT across the lanes. For vehicles that drive UP the
      screen (away from the camera), "forward" = crossing from below the line to above it.
    - A vehicle is flagged when it was BEHIND the line when the light turned red and is
      PAST it (while still red). The red-onset frame is back-dated, so a vehicle that
      crosses a split second after the light changes is no longer missed.
    - The line is treated as a segment (+20% each side) so cross-traffic that merely passes
      the extension of the line is ignored.

Wrong-way config (coordinates are in the RESIZED frame, max width 1280; FIXED camera only):
  Option A (env):
      WRONG_WAY="x1,y1,x2,y2,dx,dy"       zone box + the ALLOWED travel direction (";" for several zones)
                                          direction is in image coords: up the screen = 0,-1,
                                          down = 0,1, left->right = 1,0
      WRONG_WAY_DEBUG=true                print the direction reading of every vehicle inside a zone
  Option B (per video, WebSocket query param — overrides the env value):
      /ws/detect/video?wrong_way=x1,y1,x2,y2,dx,dy
  Tips:
    - Draw each zone only over lanes that have ONE fixed direction (a two-way road needs one
      zone per carriageway), and keep zones on straight sections (turning cars look "wrong").
    - Quick test: set the direction OPPOSITE to real traffic — normal vehicles should then be flagged.
"""

from __future__ import annotations

import sys
import os
# Ensure project root is on sys.path so `from src.X import Y` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import base64
import json
import math
import re
import tempfile
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


import cv2
import numpy as np
from fastapi import (
    FastAPI, File, HTTPException, Query, UploadFile,
    WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Gridlock — Traffic Violation Detection API",
    version="1.0.0",
    description="YOLOv8 + PaddleOCR violation detection REST and WebSocket API",
)



_origins = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:5174",   # Vite fallback port
    "http://localhost:3000",   # CRA / alt dev server
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:3000",
]
_frontend_url = os.getenv("FRONTEND_URL")
if _frontend_url:
    _origins.append(_frontend_url)
if os.getenv("CORS_ALLOW_ALL") == "true":
    _origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── GPU auto-detection ───────────────────────────────────────────────────────

try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _DEVICE = "cpu"

print(f"[api/main.py] Compute device: {_DEVICE.upper()}")

# ─── Two-stage helmet pipeline constants ──────────────────────────────────────

# Helmet model: local best.pt (Abs6187 model, classes: accept-Helmet- / non-Helmet-)
# First existing file wins. Set HELMET_MODEL=path/to/best.pt to force one.
_HELMET_CANDIDATES = [
    os.getenv("HELMET_MODEL", ""),
    str(Path(__file__).resolve().parent / "best.pt"),            # api/best.pt  <- yours
    str(Path(__file__).resolve().parent.parent / "best.pt"),     # gridlock-prototype/best.pt
    str(Path(__file__).resolve().parent.parent / "models" / "best.pt"),
]

# ── Helmet tuning (v2) ──
# Philosophy: a false fine is worse than a late one, so NO_HELMET must be confident and
# repeated, while a HELMET reading is easy to accept (it protects the rider).
_HELMET_CONF         = 0.35   # a HELMET reading needs at least this confidence
_HELMET_NO_CONF      = 0.30   # lower for small/distant bare-head detections; temporal voting still guards false alerts
_HELMET_IMGSZ        = 640    # helmet model input size (rider crops are fed at natural scale)
_HELMET_FEET_MIN     = -0.05  # helmet checks: rider feet may sit this far ABOVE the bike top
_HELMET_MOTO_ONLY    = os.getenv("HELMET_INCLUDE_BICYCLE", "").lower() not in ("1", "true", "yes")
# Set HELMET_DEBUG=true to see per-rider readings, vote counts (H:no/yes) and why alerts are skipped
_HELMET_DEBUG = os.getenv("HELMET_DEBUG", "").lower() in ("1", "true", "yes")
_MIN_VEH_PX          = 40
_HEAD_PAD            = 0.35   # 35% extra upward padding on crop (used by triple-riding evidence crop)
_RIDER_MIN_H         = 40     # rider box must be at least this tall (px) to judge the head
_MIN_RIDER_PX        = 30     # ...and at least this wide
_HEAD_ZONE           = 0.58   # expanded upper-body zone helps high-angle/seated riders
_HELMET_MIN_VOTES    = 3      # readable frames needed before judging
_HELMET_MIN_NO_VOTES = 2      # at least 2 separate no-helmet readings
_HELMET_NO_RATIO     = 0.60   # weighted majority of no-helmet evidence
_HELMET_NO_MEAN_CONF = 0.35   # allow weaker but repeated detections on small heads
_HELMET_CLEAR_VOTES  = 8      # only clear helmet evidence can close a track
_HELMET_MAX_VOTES    = 14     # undecided tracks reset their vote window and keep checking
_DEDUPE_FRAMES       = 60     # same spot within this many frames = same rider, log once

# ── COCO detector tuning (FIX) ──
_COCO_CONF   = 0.25   # was 0.30 — small / partly hidden riders (kids) need a lower bar
_COCO_IOU    = 0.65   # was 0.45 — NMS was MERGING riders sitting close together into 1-2 boxes
_COCO_IMGSZ  = 960    # was 640  — more pixels = riders on a bike separate better
# FIX (tall / portrait video): a 1280x2275 frame squeezed into imgsz=960 shrinks everything 2.4x,
# so a far-away bike at a junction becomes ~25 px and the detector loses it. For very tall/wide
# frames we now use a larger imgsz (up to COCO_IMGSZ env, default 1280). Normal 16:9 video is unchanged.
_COCO_IMGSZ_MAX = int(os.getenv("COCO_IMGSZ", "1280"))
_PERSON_DEDUPE_IOU = 0.60   # drop overlapping person boxes (same person detected twice) — was 0.85
_BIKE_DEDUPE_IOU   = 0.50   # same bike detected twice (e.g. as "motorcycle" AND "bicycle")
_VEH_DEDUPE_IOU    = 0.70   # same car detected twice (e.g. as "car" AND "truck")

# ── Triple-riding tuning (FIX) ──
_TRIPLE_WINDOW       = 10     # look at the last N processed frames of each bike...
_TRIPLE_MIN_HITS     = 5      # ...and flag if 3+ riders were seen in at least this many of them (was 3 — too easy)
                              # (NOT "in a row" any more — one bad frame no longer resets the count)
_RIDER_MIN_ON_BIKE   = 0.50   # fraction of a PERSON's box that must sit inside the bike's rider zone (was 0.30)
_RIDER_ZONE_UP       = 1.00   # rider zone extends this many bike-heights ABOVE the bike box
_RIDER_ZONE_SIDE     = 0.05   # and this fraction of bike width to each side (was 0.20 — grabbed neighbours)
_RIDER_ZONE_DOWN     = 0.10   # and a little below
_RIDER_MIN_H_RATIO   = 0.60   # a real rider's box is at least this tall vs the bike box (drops head/torso fragments)
_RIDER_MAX_H_RATIO   = 3.00   # ...and at most this tall (drops big near-camera pedestrians)
_RIDER_FEET_MIN      = 0.05   # rider's feet (box bottom) must be at least this far below the TOP of the bike box
                              # (people sitting on a truck / standing behind a bike have feet ABOVE the bike -> not riders)
_RIDER_FEET_MAX      = 0.20   # ...and at most this far below the BOTTOM of the bike box (fraction of bike height)
# Same-rider filter for MOVING cameras only (OFF by default: with a static camera the tracker
# already handles identity, and this colour filter wrongly merged different riders in dark clothes).
_LOOK_DEDUPE_ON      = os.getenv("LOOK_DEDUPE", "").lower() in ("1", "true", "yes")
_LOOK_DEDUPE_SIM     = 0.85   # colour-layout similarity (0..1) above which two NO_HELMET riders are "the same person"
_LOOK_DEDUPE_FRAMES  = 120    # ...if logged within this many frames
_RIDER_MERGE_IOA     = 0.85   # two person boxes this overlapped (vs the smaller one)...
_RIDER_MERGE_CX      = 0.25   # ...AND with centres this close (fraction of width) = ONE person detected twice
_TRIPLE_MIN_BIKE_PX  = 50     # don't judge triple riding on tiny far-away bikes (too noisy)
# Set TRIPLE_DEBUG=true to see "R:n" and the counted rider boxes on every bike
_TRIPLE_DEBUG = os.getenv("TRIPLE_DEBUG", "").lower() in ("1", "true", "yes")

# ── Track re-linking (FIX for ID switches) ──
_RELINK_MAX_GAP = 45         # a lost track can be re-claimed for this many frames
_RELINK_DIST    = 1.5        # max centre distance (in bike sizes) to treat a "new" bike as the old one

# ── Signal jump (red light) tuning ──
_SIGNAL_MIN_PX      = 30     # min coloured pixels inside the signal box(es) to trust a reading
_SIGNAL_CONFIRM_N   = 2      # FIX: a new light colour must be read this many times IN A ROW to be trusted
                             #      (was a 5-reading majority vote = ~0.3 s of lag, so a vehicle crossing right
                             #       after the light changed was judged against the OLD colour and missed)
_SIGNAL_UNKNOWN_N   = 45     # unreadable this many times in a row -> state becomes UNKNOWN
_SIGNAL_MIN_VEH_PX  = 20     # FIX: signal tracking works on smaller vehicles than helmet checks (was 40 -> far junctions skipped)
_SIGNAL_HIST_N      = 120    # processed-frame positions remembered per vehicle
_SIGNAL_MARGIN_PX   = 3.0    # must be at least this far past the line (kills box jitter)...
_SIGNAL_MARGIN_H    = 0.12   # ...or this fraction of the vehicle height, whichever is larger
_SIGNAL_SPAN_PAD    = 0.20   # the stop line is a segment, extended by this fraction each side
_SIGNAL_NEAR_PX     = 150    # SIGNAL_DEBUG prints vehicles closer than this to a line
# Set SIGNAL_DEBUG=true to print signal state changes and every vehicle near a stop line
_SIGNAL_DEBUG = os.getenv("SIGNAL_DEBUG", "").lower() in ("1", "true", "yes")

# ── Wrong-way tuning ──
# WRONG_WAY="x1,y1,x2,y2,dx,dy"  zone box + allowed travel direction (";" for several zones)
#   direction is in image coords: up the screen = 0,-1   down = 0,1   left->right = 1,0
# or per video: /ws/detect/video?wrong_way=x1,y1,x2,y2,dx,dy
_WW_WINDOW      = 12     # processed frames used to measure which way a vehicle is travelling
_WW_MIN_MOVE_PX = 30     # must travel at least this far over the window (ignores parked / jittering boxes)
_WW_MIN_MOVE_H  = 0.5    # ...or this fraction of its own height, whichever is larger
_WW_COS_MAX     = -0.5   # moving more than 120 degrees AGAINST the allowed direction
_WW_CONFIRM_N   = 4      # this many checks in a row must agree before flagging
_WW_MIN_VEH_PX  = 30
# Set WRONG_WAY_DEBUG=true to print the direction reading of every vehicle inside a zone
_WW_DEBUG = os.getenv("WRONG_WAY_DEBUG", "").lower() in ("1", "true", "yes")


def _parse_box(s: Optional[str], n: int) -> Optional[List[int]]:
    """'10,20,30,40' -> [10, 20, 30, 40]; returns None if missing / malformed."""
    if not s:
        return None
    try:
        vals = [int(float(x)) for x in s.split(",")]
        return vals if len(vals) == n else None
    except Exception:
        return None


def _parse_boxes(s: Optional[str], n: int) -> List[List[int]]:
    """'1,2,3,4;5,6,7,8' (or '|' separated) -> [[1,2,3,4],[5,6,7,8]]; bad parts are dropped."""
    if not s:
        return []
    out: List[List[int]] = []
    for part in s.replace("|", ";").split(";"):
        b = _parse_box(part.strip(), n)
        if b:
            out.append(b)
    return out


def _signal_cfg_from_env() -> dict:
    """Fresh signal config from the environment (query params then override it per video)."""
    d = os.getenv("SIGNAL_DIR", "").strip().lower()
    if d not in ("any", "forward", "reverse"):
        flip = os.getenv("SIGNAL_FLIP", "").strip().lower()
        if flip in ("1", "true", "yes"):
            d = "reverse"
        elif flip in ("0", "false", "no"):
            d = "forward"
        else:
            d = "any"      # FIX: unknown direction no longer silently means "forward only"
    return {
        "rois":      _parse_boxes(os.getenv("SIGNAL_ROI"), 4),   # [[x1,y1,x2,y2], ...] around the traffic light(s)
        "lines":     _parse_boxes(os.getenv("STOP_LINE"), 4),    # [[x1,y1,x2,y2], ...] stop line(s)
        "dir":       d,                                          # any | forward | reverse
        "state":     "UNKNOWN",                                  # RED / YELLOW / GREEN / UNKNOWN (confirmed)
        "red_onset": None,                                       # processed frame where the current RED began
    }


# Active signal config for the CURRENT video. Reloaded from env at the start of every video.
_signal_cfg: dict = _signal_cfg_from_env()

# Wrong-way config + per-vehicle state (zones are reloaded at the start of every video)
_ww_cfg    : dict = {"rules": _parse_boxes(os.getenv("WRONG_WAY"), 6)}   # [[x1,y1,x2,y2,dx,dy], ...]
_ww_hist   : dict = {}   # {vehicle_id: deque of bottom-centre points}
_ww_streak : dict = {}   # {vehicle_id: consecutive "wrong way" readings}
_ww_seen   : dict = {}   # {vehicle_id: last frame seen}

# COCO class IDs
_CLS_PERSON = 0; _CLS_BICYCLE = 1; _CLS_CAR = 2
_CLS_MOTO   = 3; _CLS_BUS     = 5; _CLS_TRUCK = 7
_BIKE_CLASSES  = {_CLS_MOTO, _CLS_BICYCLE}
_MOTOR_CLASSES = {_CLS_CAR, _CLS_BUS, _CLS_TRUCK}
_ALL_CLASSES   = [_CLS_PERSON, _CLS_BICYCLE, _CLS_CAR, _CLS_MOTO, _CLS_BUS, _CLS_TRUCK]
_CLS_LABEL     = {0:"PERSON",1:"BICYCLE",2:"CAR",3:"MOTO",5:"BUS",7:"TRUCK"}

_COLORS = {
    "NO_HELMET":     (0,  80,  255),
    "TRIPLE_RIDING": (0,  0,   255),
    "SIGNAL_JUMP":   (255, 0,  255),
    "WRONG_WAY":     (255, 200, 0),
    "OK":            (0,  200, 0),
    "SCAN":          (160,160, 0),
    "SMALL":         (60, 60,  60),
}

_SIGNAL_STATE_COLORS = {
    "RED":     (0,   0,   255),
    "YELLOW":  (0,   255, 255),
    "GREEN":   (0,   200, 0),
    "UNKNOWN": (150, 150, 150),
}

JUDGE_SERVER = os.getenv("JUDGE_SERVER", "http://localhost:8001/api/violation")

# ─── Lazy-loaded singletons ───────────────────────────────────────────────────

_detector      = None
_alpr          = None
_coco_model    = None   # YOLO COCO model for two-stage pipeline
_helmet_model  = None   # helmet YOLO model
_no_helmet_ids : set = set()
_helmet_ids    : set = set()
_triple_hist   : dict = {}   # {bike_id: deque of 0/1 — was 3+ riders seen on this processed frame?}
_triple_conf   : dict = {}   # {bike_id: avg rider confidence from the last frame with 3+ riders}
_helmet_votes  : dict = {}   # {bike_id: {"no","yes" counts, "no_w","yes_w" conf-weighted, "best" crop, "best_conf"}}
_recent_no_helmet: list = []  # [(frame_id, box)] for de-duplicating ID switches
_alias         : dict = {}   # {raw tracker id: canonical bike id} — merges ID switches
_track_last    : dict = {}   # {canonical bike id: (frame_id, box)} — last place each bike was seen
_veh_alias     : dict = {}   # FIX: same ID-switch merging for cars / buses / trucks (signal jump)
_veh_track_last: dict = {}
_signal_side   : dict = {}   # {vehicle_id: deque[(frame_id, [signed distance to each stop line])]}
_signal_rt     : dict = {"pending": None, "n": 0, "first": 0, "unknown": 0}   # light-colour state machine
_recent_no_helmet_look: list = []   # [(frame_id, colour descriptor)] of riders already logged as NO_HELMET
_video_lock: Optional[asyncio.Lock] = None   # one video at a time (all tracking state above is global)


def get_detector(conf: float = 0.45) -> Any:
    """Return the shared detector singleton, loading on first call."""
    global _detector
    if _detector is None:
        from src.detect import GridlockDetector
        _detector = GridlockDetector(conf_threshold=conf)
    return _detector


def get_alpr() -> Any:
    """Return the shared ALPR singleton, loading on first call."""
    global _alpr
    if _alpr is None:
        try:
            from src.alpr import ALPRPipeline
            _alpr = ALPRPipeline()
        except ImportError as e:
            raise HTTPException(
                status_code=500,
                detail=f"ALPR deps missing (pip install paddleocr paddlepaddle): {e}",
            )
    return _alpr


def get_checker(
    overlap: float = 0.30,
    triple: int = 3,
    signal_roi: Optional[List[int]] = None,
) -> Any:
    """ViolationChecker is lightweight — recreate per request with user params."""
    from src.violations import ViolationChecker
    return ViolationChecker(
        overlap_threshold=overlap,
        triple_riding_threshold=triple,
        signal_roi=signal_roi,
    )


# ─── Two-stage pipeline helpers ───────────────────────────────────────────────

def _ensure_helmet_weights() -> Optional[Path]:
    """Use the local best.pt (no download)."""
    for c in _HELMET_CANDIDATES:
        if c and Path(c).is_file() and Path(c).stat().st_size > 1_000_000:
            print(f"[api] Using helmet weights: {c}")
            return Path(c)
    print("[api][WARN] best.pt not found. Looked in:\n  " + "\n  ".join(c for c in _HELMET_CANDIDATES if c))
    return None

def _get_coco_model():
    """Singleton YOLO COCO model for the two-stage pipeline."""
    global _coco_model
    if _coco_model is None:
        from ultralytics import YOLO
        print("[api] Loading COCO tracking model (yolov8s.pt)…")
        _coco_model = YOLO("yolov8s.pt")
        _coco_model.to(_DEVICE)
        print(f"[api] COCO model ready on {_DEVICE.upper()}")
    return _coco_model


def _norm_name(name: str) -> str:
    """'No Helmet' / 'no_helmet' / 'NoHelmet' -> 'nohelmet' (letters only, lowercase)."""
    return re.sub(r"[^a-z]", "", str(name).lower())


def _is_no_helmet_name(name: str) -> bool:
    n = _norm_name(name)
    return any(k in n for k in ("nohelmet", "withouthelmet", "nothelmet",
                                "nonhelmet", "nohelm", "without", "bare"))


def _is_helmet_name(name: str) -> bool:
    return "helmet" in _norm_name(name) and not _is_no_helmet_name(name)


def _get_helmet_model():
    """Singleton helmet YOLO model."""
    global _helmet_model, _no_helmet_ids, _helmet_ids
    if _helmet_model is None:
        weights = _ensure_helmet_weights()
        if weights is None:
            print("[api][WARN] No helmet weights — helmet checks will be skipped.")
            return None
        try:
            from ultralytics import YOLO
            print(f"[api] Loading helmet model: {weights}")
            _helmet_model = YOLO(str(weights))
            _helmet_model.to(_DEVICE)
            _no_helmet_ids = {cid for cid, nm in _helmet_model.names.items() if _is_no_helmet_name(nm)}
            _helmet_ids    = {cid for cid, nm in _helmet_model.names.items() if _is_helmet_name(nm)}
            print(f"[api] Helmet model classes: {_helmet_model.names}")
            print(f"[api] Helmet IDs={_helmet_ids}  No-helmet IDs={_no_helmet_ids}")
            if not _helmet_ids or not _no_helmet_ids:
                print("[api][WARN] Could not map BOTH helmet and no-helmet classes — "
                      "check the class list above; helmet checks will be unreliable.")
        except Exception as e:
            print(f"[api][WARN] Could not load helmet model: {e}")
            _helmet_model = None
    return _helmet_model


def _reset_tracking_state(model=None) -> None:
    """Fresh state for every new video (counters, votes, ID aliases AND the YOLO tracker itself)."""
    _ww_hist.clear()         # wrong-way: forget last video's vehicle positions
    _ww_streak.clear()
    _ww_seen.clear()
    _triple_hist.clear()
    _triple_conf.clear()
    _helmet_votes.clear()
    _recent_no_helmet.clear()
    _recent_no_helmet_look.clear()
    _alias.clear()
    _track_last.clear()
    _veh_alias.clear()
    _veh_track_last.clear()
    _signal_side.clear()     # signal-jump: forget last video's vehicle positions
    _signal_rt.update({"pending": None, "n": 0, "first": 0, "unknown": 0})
    _signal_cfg["state"] = "UNKNOWN"      # NOTE: the ROI / lines / direction are kept — they belong to this video
    _signal_cfg["red_onset"] = None
    # FIX: the COCO model is a singleton and track(persist=True) keeps the ByteTrack
    # state alive between videos. Dropping the predictor forces a brand-new tracker,
    # so IDs from the previous video can't leak into this one.
    if model is not None:
        try:
            model.predictor = None
        except Exception:
            pass


def _pick_imgsz(frame: np.ndarray) -> int:
    """YOLO input size. Normal frames keep _COCO_IMGSZ; very tall/wide frames get more pixels."""
    long_side = max(frame.shape[:2])
    if long_side > 1.5 * _COCO_IMGSZ:
        return int(math.ceil(min(long_side, _COCO_IMGSZ_MAX) / 32) * 32)
    return _COCO_IMGSZ


# ─── Geometry / detection helpers ─────────────────────────────────────────────

def _boxes_overlap(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def _dedupe_dets(dets: list, iou_thr: float) -> list:
    """Keep the highest-confidence box among near-identical ones."""
    kept: list = []
    for d in sorted(dets, key=lambda x: -x["conf"]):
        if all(_iou(d["box"], k["box"]) < iou_thr for k in kept):
            kept.append(d)
    return kept


def _dedupe_persons(persons: list) -> list:
    """
    One rider often produces 2-3 person boxes (full body + upper body + a
    fragment). Because the COCO NMS IoU is loose, those survive and get counted
    as extra riders -> single riders flagged as TRIPLE.
    A box is a duplicate of a stronger one if:
      - IoU is high, OR
      - it is almost fully INSIDE the other box (IoA) AND sits in the same column
        (centres close). Real pillion riders sit side-by-side / offset, so they survive.
    """
    kept: list = []
    for p in sorted(persons, key=lambda x: -x["conf"]):
        px1, py1, px2, py2 = p["box"]
        pw = max(px2 - px1, 1.0)
        pa = max(pw * (py2 - py1), 1.0)
        pcx = (px1 + px2) / 2
        is_dup = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k["box"]
            kw = max(kx2 - kx1, 1.0)
            ka = max(kw * (ky2 - ky1), 1.0)
            ix = max(0.0, min(px2, kx2) - max(px1, kx1))
            iy = max(0.0, min(py2, ky2) - max(py1, ky1))
            ioa = (ix * iy) / min(pa, ka)
            same_col = abs(pcx - (kx1 + kx2) / 2) < _RIDER_MERGE_CX * max(pw, kw)
            if _iou(p["box"], k["box"]) >= _PERSON_DEDUPE_IOU or (ioa >= _RIDER_MERGE_IOA and same_col):
                is_dup = True
                break
        if not is_dup:
            kept.append(p)
    return kept


def _resolve_bike_ids(bikes: list, frame_id: int,
                      alias: Optional[dict] = None,
                      track_last: Optional[dict] = None) -> None:
    """
    FIX (duplicate counting): ByteTrack hands out a NEW id when it loses a fast bike
    for a moment. Every new id restarted the helmet votes and could log the same bike
    again. Here a 'new' id that appears close to a bike that just disappeared is
    re-linked to the old (canonical) id, so votes / confirmations carry over.
    Writes the canonical id back into each bike's det["id"].

    FIX: also used for cars / buses / trucks (pass _veh_alias / _veh_track_last) so a
    car whose tracker id switches is still the SAME vehicle for signal-jump checks.
    """
    alias      = _alias      if alias      is None else alias
    track_last = _track_last if track_last is None else track_last
    present: set = set()
    pending: list = []

    # Pass 1: ids we already know (alias or first-seen)
    for b in bikes:
        raw = b["id"]
        if raw < 0:
            continue
        if raw in alias:
            canon = alias[raw]
            if canon in present:          # two raw ids mapped to one canon in the same frame
                canon = raw               # keep them apart rather than merge two real bikes
            b["id"] = canon
            present.add(canon)
        else:
            pending.append(b)

    # Pass 2: brand-new raw ids — try to re-link to a recently lost bike
    for b in pending:
        raw = b["id"]
        x1, y1, x2, y2 = b["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        size = max(x2 - x1, y2 - y1, 1.0)
        best, best_d = None, None
        for canon, (fid, lb) in track_last.items():
            if canon in present or frame_id - fid > _RELINK_MAX_GAP:
                continue
            lcx, lcy = (lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2
            lsize = max(lb[2] - lb[0], lb[3] - lb[1], 1.0)
            d = math.hypot(cx - lcx, cy - lcy) / max(size, lsize)
            ratio = size / lsize
            if d <= _RELINK_DIST and 0.5 <= ratio <= 2.0 and (best_d is None or d < best_d):
                best, best_d = canon, d
        canon = best if best is not None else raw
        alias[raw] = canon
        b["id"] = canon
        present.add(canon)

    for b in bikes:
        if b["id"] >= 0:
            track_last[b["id"]] = (frame_id, b["box"])

    # keep the dict small
    for k in [k for k, (f, _) in track_last.items() if frame_id - f > 600]:
        del track_last[k]


def _assign_riders(bikes: list, persons: list, loose: bool = False) -> dict:
    """Associate person detections with bikes using geometry, with a tolerant helmet mode.

    Strict mode is used for triple-riding counts. Loose mode is used for helmet checks,
    where partial/leaning passengers must not be discarded just because their box centre
    sits slightly outside the bike box. Each person is assigned to at most one bike.
    """
    riders = {i: [] for i in range(len(bikes))}
    if not bikes or not persons:
        return riders

    zones = []
    for b in bikes:
        x1, y1, x2, y2 = map(float, b["box"])
        bw, bh = max(1.0, x2-x1), max(1.0, y2-y1)
        side = 0.12 if loose else _RIDER_ZONE_SIDE
        up = 1.30 if loose else _RIDER_ZONE_UP
        down = 0.18 if loose else _RIDER_ZONE_DOWN
        zones.append((x1-side*bw, y1-up*bh, x2+side*bw, y2+down*bh,
                      bh, y1, y2, (x1+x2)/2, bw))

    for person in persons:
        px1, py1, px2, py2 = map(float, person["box"])
        pw, ph = max(1.0, px2-px1), max(1.0, py2-py1)
        pcx, pcy = (px1+px2)/2, (py1+py2)/2
        parea = pw*ph
        best_i, best_score = None, -1.0
        for i, (zx1, zy1, zx2, zy2, bike_h, bike_y1, bike_y2, bike_cx, bike_w) in enumerate(zones):
            h_ratio = ph / max(bike_h, 1.0)
            h_lo, h_hi = ((0.60, 4.0) if loose else (_RIDER_MIN_H_RATIO, _RIDER_MAX_H_RATIO))
            if not h_lo <= h_ratio <= h_hi:
                continue
            # Feet should reach the bike's vertical area. Allow more tolerance for seated
            # riders on high-angle footage, but reject people far above/alongside the bike.
            feet_min = bike_y1 - (0.35 if loose else _RIDER_FEET_MIN) * bike_h
            feet_max = bike_y2 + (0.25 if loose else _RIDER_FEET_MAX) * bike_h
            if not feet_min <= py2 <= feet_max:
                continue
            ix = max(0.0, min(px2, zx2)-max(px1, zx1))
            iy = max(0.0, min(py2, zy2)-max(py1, zy1))
            frac = ix*iy/parea
            centre_inside = zx1 <= pcx <= zx2 and zy1 <= pcy <= zy2
            # In loose mode, overlap can substitute for centre-in-zone when a passenger leans.
            if loose:
                if not centre_inside and frac < 0.06:
                    continue
                horizontal_distance = abs(pcx-bike_cx)/max(bike_w, 1.0)
                if horizontal_distance > 0.90:
                    continue
                # COCO person boxes can cover only the torso on crowded/high-angle footage.
                # Keep weak-but-plausible associations for helmet checks.
                score = frac - 0.10*horizontal_distance + (0.05 if centre_inside else 0.0)
                min_score = 0.10
            else:
                if not centre_inside:
                    continue
                score = frac - 0.3*abs(pcx-(zx1+zx2)/2)/max(zx2-zx1, 1.0)
                min_score = _RIDER_MIN_ON_BIKE
            if score > best_score:
                best_i, best_score = i, score
        if best_i is not None and best_score >= min_score:
            riders[best_i].append(person)
    return riders


def _crop_head(frame: np.ndarray, box: list) -> np.ndarray:
    h, w = frame.shape[:2]
    bh = box[3] - box[1]
    top_ex = int(bh * _HEAD_PAD)
    x1 = max(0, int(box[0]) - 20)
    y1 = max(0, int(box[1]) - 20 - top_ex)
    x2 = min(w, int(box[2]) + 20)
    y2 = min(h, int(box[3]) + 20)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame[:10, :10].copy()
    # Upscale small crops
    ch, cw = crop.shape[:2]
    if min(ch, cw) < 224:
        scale = 224 / min(ch, cw)
        crop = cv2.resize(crop, (max(int(cw*scale), 224), max(int(ch*scale), 224)),
                          interpolation=cv2.INTER_LANCZOS4)
    return crop.copy()


def _head_reading(frame: np.ndarray, pbox: list):
    """Classify one rider's head area; return SKIP when evidence is not clear enough."""
    hmodel = _get_helmet_model()
    if hmodel is None:
        return "SKIP", None, 0.0
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(float, pbox)
    w, h = x2-x1, y2-y1
    if h < _RIDER_MIN_H or w < _MIN_RIDER_PX:
        if _HELMET_DEBUG:
            print(f"[helmet] SKIP small person box w={w:.0f} h={h:.0f}")
        return "SKIP", None, 0.0

    # Include the whole head and shoulders, but not most of the motorcycle/background.
    cx1, cx2 = max(0, int(x1-0.6*w)), min(W, int(x2+0.6*w))
    cy1, cy2 = max(0, int(y1-0.4*h)), min(H, int(y2+0.1*h))
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return "SKIP", None, 0.0

    # Upscale only small rider crops; scale predictions back before checking their position.
    ch, cw = crop.shape[:2]
    scale = min(2.0, max(1.0, 320.0/max(1, min(ch, cw))))
    model_crop = cv2.resize(crop, (int(cw*scale), int(ch*scale)), interpolation=cv2.INTER_CUBIC) if scale > 1.01 else crop
    try:
        res = hmodel.predict(model_crop, conf=0.20, imgsz=_HELMET_IMGSZ,
                             device=_DEVICE, verbose=False)
    except Exception as e:
        print(f"[helmet][WARN] inference failed: {e}")
        return "SKIP", None, 0.0
    
    if not res or res[0].boxes is None or len(res[0].boxes) == 0:
        if _HELMET_DEBUG:
            print(
                f"[helmet][RAW] no raw boxes; "
                f"person={int(x1)},{int(y1)},{int(x2)},{int(y2)}"
            )
        return "SKIP", None, 0.0

    if _HELMET_DEBUG:
        raw_boxes = res[0].boxes
        raw_classes = raw_boxes.cls.cpu().numpy().astype(int)
        raw_confs = raw_boxes.conf.cpu().numpy()
        print(
            f"[helmet][RAW] person={int(x1)},{int(y1)},{int(x2)},{int(y2)} "
            f"predictions="
            f"{[(hmodel.names.get(int(c), str(int(c))), round(float(cf), 3)) for c, cf in zip(raw_classes, raw_confs)]}"
        )

    xyxy = res[0].boxes.xyxy.cpu().numpy() / scale

    clses = res[0].boxes.cls.cpu().numpy().astype(int)
    confs = res[0].boxes.conf.cpu().numpy()
    # Only accept boxes substantially overlapping the rider's head zone.
    head_x1, head_x2 = x1-0.25*w, x2+0.25*w
    head_y1, head_y2 = y1-0.18*h, y1+0.60*h
    best_no = best_yes = None
    for (bx1, by1, bx2, by2), cls_id, conf in zip(xyxy, clses, confs):
        is_no = int(cls_id) in _no_helmet_ids
        is_yes = int(cls_id) in _helmet_ids
        if not (is_no or is_yes):
            continue
        gx1, gy1 = bx1+cx1, by1+cy1
        gx2, gy2 = bx2+cx1, by2+cy1
        gcx, gcy = (gx1+gx2)/2, (gy1+gy2)/2
        box_h = max(0.0, gy2-gy1)
        if not (head_x1 <= gcx <= head_x2 and head_y1 <= gcy <= head_y2):
            if _HELMET_DEBUG:
                print("[helmet] FILTERED: detection center outside head region")
            continue

        if box_h <= 1 or box_h > 0.65*h:
            if _HELMET_DEBUG:
                print(f"[helmet] FILTERED: invalid box height={box_h:.1f}, rider height={h:.1f}")
            continue
        # The detection must overlap the expected head region, not merely have its centre nearby.
        overlap_w = max(0.0, min(gx2, head_x2)-max(gx1, head_x1))
        overlap_h = max(0.0, min(gy2, head_y2)-max(gy1, head_y1))
        if overlap_w*overlap_h < 0.08*max((gx2-gx1)*(gy2-gy1), 1.0):
            continue
        item = (float(conf), [gx1, gy1, gx2, gy2])
        if is_no and conf >= _HELMET_NO_CONF and (best_no is None or conf > best_no[0]):
            best_no = item
        if is_yes and conf >= _HELMET_CONF and (best_yes is None or conf > best_yes[0]):
            best_yes = item

    if best_no is None and best_yes is None:
        if _HELMET_DEBUG:
            print(f"[helmet] SKIP no helmet-class box in crop; person={int(x1)},{int(y1)},{int(x2)},{int(y2)}")
        return "SKIP", None, 0.0
    # If both labels compete, do not issue a no-helmet result unless it wins clearly.
    if best_no is not None and (best_yes is None or best_no[0] >= best_yes[0]+0.12):
        verdict, (cf, gb) = "NO_HELMET", best_no
    elif best_yes is not None:
        verdict, (cf, gb) = "HELMET", best_yes
    else:
        return "SKIP", None, 0.0

    pad_w, pad_h = 0.65*(gb[2]-gb[0]), 0.65*(gb[3]-gb[1])
    ex1, ex2 = max(0, int(gb[0]-pad_w)), min(W, int(gb[2]+pad_w))
    ey1, ey2 = max(0, int(gb[1]-pad_h)), min(H, int(gb[3]+pad_h))
    evidence = frame[ey1:ey2, ex1:ex2]
    if _HELMET_DEBUG:
        print(f"[helmet] {verdict} conf={cf:.3f} rider={int(x1)},{int(y1)},{int(x2)},{int(y2)}")
    return verdict, (evidence.copy() if evidence.size else crop.copy()), cf


def _helmet_frame_verdict(frame: np.ndarray, bike_box: list, riders: list):
    """
    One helmet reading for one bike on one frame.
    Any rider clearly without a helmet = NO_HELMET. No riders found = SKIP
    (empty / parked bikes and background are no longer judged).
    Returns (verdict, crop, conf).
    """
    best_no = best_yes = None
    for p in riders:
        v, cr, cf = _head_reading(frame, p["box"])
        if v == "NO_HELMET" and (best_no is None or cf > best_no[1]):
            best_no = (cr, cf)
        elif v == "HELMET" and (best_yes is None or cf > best_yes[1]):
            best_yes = (cr, cf)
    # A confident no-helmet reading from any assigned rider is relevant, but only if it
    # is meaningfully stronger than any helmet reading from another rider in the same bike.
    if best_no is not None and (best_yes is None or best_no[1] >= best_yes[1] - 0.03):
        return "NO_HELMET", best_no[0], best_no[1]
    if best_yes is not None:
        return "HELMET", best_yes[0], best_yes[1]
    if best_no is not None:
        return "SKIP", None, 0.0
    return "SKIP", None, 0.0



def _helmet_detect_full(frame: np.ndarray) -> list:
    """Run the helmet model ONCE on the whole frame (same as your standalone test)."""
    hmodel = _get_helmet_model()
    if hmodel is None:
        return []
    try:
        res = hmodel.predict(frame, conf=0.25, imgsz=1280, device=_DEVICE, verbose=False)
    except Exception as e:
        print(f"[helmet][WARN] full-frame inference failed: {e}")
        return []
    out = []
    if not res or res[0].boxes is None:
        return out
    b = res[0].boxes
    for xyxy, c, cf in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()):
        if int(c) in _no_helmet_ids:
            kind = "NO"
        elif int(c) in _helmet_ids:
            kind = "YES"
        else:
            continue
        out.append({"kind": kind, "conf": float(cf), "box": xyxy.tolist()})
    if _HELMET_DEBUG:
        print(f"[helmet][FULL] {[(d['kind'], round(d['conf'], 2)) for d in out]}")
    return out


def _helmet_verdict_for_bike(frame: np.ndarray, bike_box: list, dets: list, riders: list = None):
    """Pick the helmet / no-helmet detections that sit on THIS bike's riders."""
    x1, y1, x2, y2 = bike_box
    bw, bh = x2 - x1, y2 - y1
    zones = []
    for p in (riders or []):
        px1, py1, px2, py2 = p["box"]
        pw, ph = px2 - px1, py2 - py1
        zones.append((px1 - 0.2 * pw, py1 - 0.2 * ph, px2 + 0.2 * pw, py1 + 0.5 * ph))
    if not zones:   # no person linked: fall back to a tall zone above the bike
        zones.append((x1 - 0.25 * bw, y1 - 3.0 * bh, x2 + 0.25 * bw, y1 + 0.5 * bh))
    H, W = frame.shape[:2]
    best_no = best_yes = None
    for d in dets:
        bx1, by1, bx2, by2 = d["box"]
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        if not any(a <= cx <= c and b <= cy <= e for a, b, c, e in zones):
            continue
        if d["kind"] == "NO" and d["conf"] >= _HELMET_NO_CONF and (best_no is None or d["conf"] > best_no[0]):
            best_no = (d["conf"], d["box"])
        if d["kind"] == "YES" and d["conf"] >= _HELMET_CONF and (best_yes is None or d["conf"] > best_yes[0]):
            best_yes = (d["conf"], d["box"])
    if best_no is None and best_yes is None:
        return "SKIP", None, 0.0
    if best_no is not None and (best_yes is None or best_no[0] >= best_yes[0] + 0.05):
        cf, gb = best_no
        verdict = "NO_HELMET"
    else:
        cf, gb = best_yes
        verdict = "HELMET"
    pw, ph = 0.65 * (gb[2] - gb[0]), 0.65 * (gb[3] - gb[1])
    ev = frame[max(0, int(gb[1] - ph)):min(H, int(gb[3] + ph)),
               max(0, int(gb[0] - pw)):min(W, int(gb[2] + pw))]
    return verdict, (ev.copy() if ev.size else None), cf


def _helmet_verdict_per_rider(frame: np.ndarray, bike_box: list, dets: list, riders: list = None):
    """
    FIX: judge EACH rider on the bike separately. Any bare-headed rider = NO_HELMET, even if
    the front rider's helmet is detected with higher confidence (old code lost bare-headed pillions).
    No person linked to the bike -> SKIP (empty / parked bikes are no longer judged).
    """
    if not riders:
        return "SKIP", None, 0.0
    H, W = frame.shape[:2]
    zones = []
    for p in riders:
        px1, py1, px2, py2 = p["box"]
        pw, ph = px2 - px1, py2 - py1
        zones.append({
            "rect": (px1 - 0.2 * pw, py1 - 0.2 * ph, px2 + 0.2 * pw, py1 + 0.5 * ph),
            "head": ((px1 + px2) / 2, py1 + 0.1 * ph),
            "no": None, "yes": None,
        })
    for d in dets:
        bx1, by1, bx2, by2 = d["box"]
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        best_z, best_dist = None, None
        for z in zones:                      # each detection belongs to the NEAREST rider only
            a, b, c, e = z["rect"]
            if not (a <= cx <= c and b <= cy <= e):
                continue
            dist = math.hypot(cx - z["head"][0], cy - z["head"][1])
            if best_dist is None or dist < best_dist:
                best_z, best_dist = z, dist
        if best_z is None:
            continue
        if d["kind"] == "NO" and d["conf"] >= _HELMET_NO_CONF and (best_z["no"] is None or d["conf"] > best_z["no"][0]):
            best_z["no"] = (d["conf"], d["box"])
        if d["kind"] == "YES" and d["conf"] >= _HELMET_CONF and (best_z["yes"] is None or d["conf"] > best_z["yes"][0]):
            best_z["yes"] = (d["conf"], d["box"])

    best_no = best_yes = None
    for z in zones:
        zn, zy = z["no"], z["yes"]
        if zn is not None and (zy is None or zn[0] >= zy[0] + 0.05):
            if best_no is None or zn[0] > best_no[0]:
                best_no = zn
        elif zy is not None:
            if best_yes is None or zy[0] > best_yes[0]:
                best_yes = zy
    if best_no is not None:
        cf, gb, verdict = best_no[0], best_no[1], "NO_HELMET"
    elif best_yes is not None:
        cf, gb, verdict = best_yes[0], best_yes[1], "HELMET"
    else:
        return "SKIP", None, 0.0
    pw, ph = 0.65 * (gb[2] - gb[0]), 0.65 * (gb[3] - gb[1])
    ev = frame[max(0, int(gb[1] - ph)):min(H, int(gb[3] + ph)),
               max(0, int(gb[0] - pw)):min(W, int(gb[2] + pw))]
    return verdict, (ev.copy() if ev.size else None), cf


def _is_duplicate_no_helmet(frame_id: int, box: list) -> bool:
    """
    True if a NO_HELMET was already logged near this spot a moment ago
    (the tracker gave the same rider a new ID).

    The allowed distance GROWS with the number of frames since the earlier log (a fast bike
    travels far between processed frames) but is capped tighter than before, so different
    bikes in dense traffic no longer suppress each other.
    """
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    size = max(box[2] - box[0], box[3] - box[1], 1.0)
    _recent_no_helmet[:] = [r for r in _recent_no_helmet if frame_id - r[0] <= _DEDUPE_FRAMES]
    for fid, b in _recent_no_helmet:
        bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        bsize = max(b[2] - b[0], b[3] - b[1], 1.0)
        ratio = size / bsize
        if not (0.5 <= ratio <= 2.0):
            continue
        tol = min(0.5 + 0.02 * (frame_id - fid), 1.2)
        if math.hypot(cx - bcx, cy - bcy) < tol * max(size, bsize):
            return True
    return False


# ─── Same-rider filter (for moving cameras) ───────────────────────────────────

def _rider_descriptor(frame: np.ndarray, box: list) -> Optional[np.ndarray]:
    """
    Colour-LAYOUT fingerprint of a bike + rider: the crop is split into 3 vertical
    bands (head/torso/bike) and each band gets an HSV histogram. Position-free, so
    it still works when the camera itself is moving and tracker IDs keep changing.
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    bh = y2 - y1
    cx1, cx2 = max(0, int(x1)), min(W, int(x2))
    cy1, cy2 = max(0, int(y1 - 0.35 * bh)), min(H, int(y2))
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(cv2.resize(crop, (64, 96)), cv2.COLOR_BGR2HSV)
    parts = []
    for b in range(3):
        band = hsv[b * 32:(b + 1) * 32]
        h = cv2.calcHist([band], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
        parts.append(h / max(float(h.sum()), 1.0))
    return np.concatenate(parts) / 3.0


def _is_same_rider_look(frame_id: int, desc: np.ndarray) -> bool:
    """
    True if a rider that looks like this was already logged recently.
    On a match the stored entry's frame is refreshed, so one rider who stays in
    view for a long time keeps being recognised (chain of matches).
    """
    _recent_no_helmet_look[:] = [r for r in _recent_no_helmet_look
                                 if frame_id - r[0] <= _LOOK_DEDUPE_FRAMES]
    for i, (_fid, d) in enumerate(_recent_no_helmet_look):
        if float(np.minimum(d, desc).sum()) >= _LOOK_DEDUPE_SIM:   # histogram intersection
            _recent_no_helmet_look[i] = (frame_id, d)
            return True
    return False


# ─── Signal-jump (red light) helpers ──────────────────────────────────────────

def _signal_raw_reading(frame: np.ndarray, rois: List[List[int]]) -> str:
    """
    One raw reading of the traffic light: RED / YELLOW / GREEN / UNKNOWN, from the
    coloured pixels (HSV) inside every signal box. No smoothing here — the state
    machine in _update_signal_state() decides when a change is trusted.
    """
    H, W = frame.shape[:2]
    totals = {"RED": 0, "YELLOW": 0, "GREEN": 0}
    for roi in rois:
        x1, y1, x2, y2 = roi
        xa, xb = max(0, min(x1, x2)), min(W, max(x1, x2))
        ya, yb = max(0, min(y1, y2)), min(H, max(y1, y2))
        patch = frame[ya:yb, xa:xb]
        if patch.size == 0:
            continue
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        # red wraps around the hue axis, so it needs two ranges
        red    = cv2.inRange(hsv, (0, 120, 150),   (10, 255, 255)) | \
                 cv2.inRange(hsv, (170, 120, 150), (180, 255, 255))
        yellow = cv2.inRange(hsv, (15, 120, 150),  (35, 255, 255))
        green  = cv2.inRange(hsv, (40, 80, 120),   (90, 255, 255))
        totals["RED"]    += int(np.count_nonzero(red))
        totals["YELLOW"] += int(np.count_nonzero(yellow))
        totals["GREEN"]  += int(np.count_nonzero(green))
    best = max(totals, key=totals.get)
    return best if totals[best] >= _SIGNAL_MIN_PX else "UNKNOWN"


def _update_signal_state(raw: str, frame_id: int) -> str:
    """
    Debounced light state. A new colour is accepted after _SIGNAL_CONFIRM_N identical
    readings in a row (kills one-frame flicker) and the frame where that colour FIRST
    appeared is remembered as the onset — so for RED we know exactly when it began,
    even though we only confirm it a moment later.
    """
    rt = _signal_rt
    cur = _signal_cfg["state"]

    if raw == "UNKNOWN":
        rt["unknown"] += 1
        if rt["unknown"] >= _SIGNAL_UNKNOWN_N and cur != "UNKNOWN":
            _signal_cfg["state"], _signal_cfg["red_onset"] = "UNKNOWN", None
            if _SIGNAL_DEBUG:
                print(f"[signal] f{frame_id} light unreadable -> UNKNOWN")
        return _signal_cfg["state"]
    rt["unknown"] = 0

    # RED never goes straight to YELLOW (it goes RED -> GREEN -> YELLOW -> RED), so a YELLOW
    # reading during RED is a glare / street-lamp false reading — ignore it, otherwise it would
    # wipe the red-onset and the vehicles already behind the line would be forgotten.
    if cur == "RED" and raw == "YELLOW":
        rt["pending"], rt["n"] = None, 0
        return cur

    if raw == cur:
        rt["pending"], rt["n"] = None, 0
        return cur

    if rt["pending"] == raw:
        rt["n"] += 1
    else:
        rt["pending"], rt["n"], rt["first"] = raw, 1, frame_id

    if rt["n"] >= _SIGNAL_CONFIRM_N:
        _signal_cfg["state"] = raw
        _signal_cfg["red_onset"] = rt["first"] if raw == "RED" else None
        if _SIGNAL_DEBUG:
            print(f"[signal] f{frame_id} light {cur} -> {raw}"
                  + (f"  (red started at f{rt['first']})" if raw == "RED" else ""))
        rt["pending"], rt["n"] = None, 0
    return _signal_cfg["state"]


def _line_dist(p, ln) -> float:
    """Signed distance (pixels) from point p to the infinite line ln=[x1,y1,x2,y2]."""
    dx, dy = ln[2] - ln[0], ln[3] - ln[1]
    length = math.hypot(dx, dy) or 1.0
    return (dx * (p[1] - ln[1]) - dy * (p[0] - ln[0])) / length


def _within_span(p, ln) -> bool:
    """True if p projects onto the stop-line SEGMENT (plus a little padding each side)."""
    dx, dy = ln[2] - ln[0], ln[3] - ln[1]
    l2 = dx * dx + dy * dy
    if l2 <= 0:
        return True
    t = ((p[0] - ln[0]) * dx + (p[1] - ln[1]) * dy) / l2
    return -_SIGNAL_SPAN_PAD <= t <= 1.0 + _SIGNAL_SPAN_PAD


def _crossed_line(ref: float, cur: float, direction: str, margin: float) -> bool:
    """ref/cur are signed distances. forward = + side -> - side, reverse = the opposite."""
    fwd = ref > 0 and cur <= -margin
    rev = ref < 0 and cur >= margin
    if direction == "forward":
        return fwd
    if direction == "reverse":
        return rev
    return fwd or rev


def _check_signal_jump(
    vehicles: list,
    state: str,
    onset: Optional[int],
    lines: List[List[int]],
    direction: str,
    frame: np.ndarray,
    frame_id: int,
    ts: str,
    confirmed_ids: set,
) -> list:
    """
    A signal jump = a tracked vehicle that was BEHIND a stop line when the light turned
    red and is PAST it while the light is still red.

    FIX: instead of only comparing two consecutive frames (and only when the light was
    already confirmed RED on the first of them), we remember each vehicle's position
    history and compare its CURRENT position with where it was at the red onset. So a
    vehicle that crosses right after the change is caught even though the red was only
    confirmed a few frames later. Each vehicle is flagged once.
    """
    out: list = []
    red = (state == "RED" and onset is not None)

    for v in vehicles:
        vid = v["id"]
        if vid < 0 or v["bw"] < _SIGNAL_MIN_VEH_PX or v["bh"] < _SIGNAL_MIN_VEH_PX:
            continue
        x1, y1, x2, y2 = v["box"]
        foot = ((x1 + x2) / 2, y2)       # bottom-centre ≈ where the vehicle touches the road
        dists = [_line_dist(foot, ln) for ln in lines]
        hist = _signal_side.setdefault(vid, deque(maxlen=_SIGNAL_HIST_N))
        hist.append((frame_id, dists))

        if _SIGNAL_DEBUG and any(abs(d) < _SIGNAL_NEAR_PX for d in dists):
            print(f"[signal] f{frame_id} {state:<7} veh#{vid} {_CLS_LABEL.get(v['cls'], 'VEH')}"
                  f" dist-to-line={[round(d) for d in dists]}")

        if not red or (vid, "SIGNAL_JUMP") in confirmed_ids:
            continue

        # where was this vehicle when the light turned red? (last position BEFORE the onset;
        # if we only met it after the onset, its first known position)
        ref = hist[0][1]
        for fid, ds in hist:
            if fid < onset:
                ref = ds
            else:
                break

        margin = max(_SIGNAL_MARGIN_PX, _SIGNAL_MARGIN_H * v["bh"])
        for li, ln in enumerate(lines):
            if not _within_span(foot, ln):
                continue
            if not _crossed_line(ref[li], dists[li], direction, margin):
                continue
            confirmed_ids.add((vid, "SIGNAL_JUMP"))
            if _SIGNAL_DEBUG:
                print(f"[signal] f{frame_id} *** SIGNAL_JUMP veh#{vid} line#{li + 1} "
                      f"(at red onset {ref[li]:.0f}px -> now {dists[li]:.0f}px)")
            crop = frame[max(0, int(y1)):max(0, int(y2)), max(0, int(x1)):max(0, int(x2))]
            try:
                cb = _encode_jpeg(crop) if crop.size else b""
            except Exception:
                cb = b""
            out.append({
                "violation_type": "SIGNAL_JUMP",
                "vehicle_id": vid,
                "bbox": v["box"],
                "confidence": round(float(v["conf"]), 3),
                "frame_id": frame_id,
                "timestamp": ts,
                "plate_text": "UNKNOWN",
                "rider_count": 0,
                "_crop_bytes": cb,
            })
            break

    # forget vehicles we haven't seen for a long time
    if len(_signal_side) > 400:
        for k in [k for k, h in _signal_side.items() if h and frame_id - h[-1][0] > 900]:
            del _signal_side[k]
    return out


# ─── Wrong-way helpers ────────────────────────────────────────────────────────

def _check_wrong_way(vehicles: list, rules: List[List[int]], frame: np.ndarray,
                     frame_id: int, ts: str, confirmed_ids: set) -> list:
    """
    A wrong-way vehicle keeps moving AGAINST the allowed direction of its zone.
    We compare where the vehicle actually travelled over the last _WW_WINDOW processed
    frames with the zone's allowed direction (dot product: +1 = with traffic, -1 = head-on).
    It must be clearly against traffic _WW_CONFIRM_N checks in a row, then it is flagged once.
    """
    out: list = []
    for v in vehicles:
        vid = v["id"]
        if vid < 0 or v["bw"] < _WW_MIN_VEH_PX or v["bh"] < _WW_MIN_VEH_PX:
            continue
        x1, y1, x2, y2 = v["box"]
        foot = ((x1 + x2) / 2, y2)       # bottom-centre ~ where the vehicle touches the road
        hist = _ww_hist.setdefault(vid, deque(maxlen=_WW_WINDOW + 1))
        hist.append(foot)
        _ww_seen[vid] = frame_id
        if (vid, "WRONG_WAY") in confirmed_ids or len(hist) <= _WW_WINDOW:
            continue

        rule = next((r for r in rules
                     if min(r[0], r[2]) <= foot[0] <= max(r[0], r[2])
                     and min(r[1], r[3]) <= foot[1] <= max(r[1], r[3])), None)
        if rule is None:
            _ww_streak[vid] = 0
            continue

        ax, ay = rule[4], rule[5]
        an = math.hypot(ax, ay)
        mx, my = foot[0] - hist[0][0], foot[1] - hist[0][1]
        mn = math.hypot(mx, my)
        if an == 0 or mn < max(_WW_MIN_MOVE_PX, _WW_MIN_MOVE_H * v["bh"]):
            _ww_streak[vid] = 0          # barely moving: no direction to judge
            continue

        cos = (mx * ax + my * ay) / (mn * an)
        _ww_streak[vid] = (_ww_streak.get(vid, 0) + 1) if cos <= _WW_COS_MAX else 0
        if _WW_DEBUG:
            print(f"[wrongway] f{frame_id} veh#{vid} {_CLS_LABEL.get(v['cls'], 'VEH')} "
                  f"cos={cos:.2f} moved={mn:.0f}px streak={_ww_streak[vid]}")
        if _ww_streak[vid] < _WW_CONFIRM_N:
            continue

        confirmed_ids.add((vid, "WRONG_WAY"))
        if _WW_DEBUG:
            print(f"[wrongway] f{frame_id} *** WRONG_WAY veh#{vid}")
        crop = frame[max(0, int(y1)):max(0, int(y2)), max(0, int(x1)):max(0, int(x2))]
        try:
            cb = _encode_jpeg(crop) if crop.size else b""
        except Exception:
            cb = b""
        out.append({
            "violation_type": "WRONG_WAY",
            "vehicle_id": vid,
            "bbox": v["box"],
            "confidence": round(float(v["conf"]), 3),
            "frame_id": frame_id,
            "timestamp": ts,
            "plate_text": "UNKNOWN",
            "rider_count": 0,
            "_crop_bytes": cb,
        })

    # forget vehicles we haven't seen for a long time
    if len(_ww_hist) > 400:
        for k in [k for k, f in _ww_seen.items() if frame_id - f > 900]:
            _ww_hist.pop(k, None)
            _ww_streak.pop(k, None)
            _ww_seen.pop(k, None)
    return out


def _draw_wrong_way_overlay(disp: np.ndarray) -> None:
    """Draw each wrong-way zone and an arrow showing the ALLOWED direction."""
    for x1, y1, x2, y2, ax, ay in _ww_cfg.get("rules") or []:
        col = _COLORS["WRONG_WAY"]
        cv2.rectangle(disp, (x1, y1), (x2, y2), col, 1)
        n = math.hypot(ax, ay) or 1.0
        cx, cy, L = (x1 + x2) // 2, (y1 + y2) // 2, 40
        cv2.arrowedLine(disp, (int(cx - ax / n * L), int(cy - ay / n * L)),
                        (int(cx + ax / n * L), int(cy + ay / n * L)), col, 3, tipLength=0.35)
        cv2.putText(disp, "ALLOWED DIRECTION", (min(x1, x2), max(14, min(y1, y2) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)


def _draw_signal_overlay(disp: np.ndarray) -> None:
    """Draw the stop line(s), the signal box(es) and the current signal state on the frame."""
    lines, rois = _signal_cfg.get("lines") or [], _signal_cfg.get("rois") or []
    state = _signal_cfg.get("state", "UNKNOWN")
    col = _SIGNAL_STATE_COLORS.get(state, _SIGNAL_STATE_COLORS["UNKNOWN"])
    for i, line in enumerate(lines):
        cv2.line(disp, (line[0], line[1]), (line[2], line[3]), _COLORS["SIGNAL_JUMP"], 2)
        cv2.putText(disp, f"STOP LINE {i + 1}" if len(lines) > 1 else "STOP LINE",
                    (line[0], max(14, line[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLORS["SIGNAL_JUMP"], 1)
    for i, roi in enumerate(rois):
        cv2.rectangle(disp, (roi[0], roi[1]), (roi[2], roi[3]), col, 2)
        if i == 0:
            cv2.putText(disp, f"SIGNAL: {state}", (roi[0], max(14, roi[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(disp, f"SIGNAL: {state}", (roi[0], max(14, roi[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
    if bool(rois) != bool(lines):     # only half configured -> the check is silently OFF; say so on screen
        msg = "SIGNAL CHECK OFF: need BOTH signal_roi and stop_line"
        cv2.putText(disp, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(disp, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)


def _encode_jpeg(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


def _annotate(
    frame: np.ndarray,
    bikes: list,
    persons: list,
    motor_vehs: list,
    violations: list,
    known_violations: Optional[dict] = None,
) -> np.ndarray:
    """
    Draw bounding boxes + labels on frame.
    `known_violations` is a persistent dict {vehicle_id: violation_type} that
    accumulates across all frames so previously confirmed vehicles keep their
    highlight even on frames where they emit no new violation event.
    """
    disp = frame.copy()
    # Merge this frame's new violations into the running tally
    if known_violations is not None:
        for v in violations:
            known_violations[v["vehicle_id"]] = v["violation_type"]
    # Build per-frame lookup — use known_violations when available
    viol_types = known_violations if known_violations is not None else \
                 {v["vehicle_id"]: v["violation_type"] for v in violations}

    for p in persons:
        x1,y1,x2,y2 = [int(v) for v in p["box"]]
        cv2.rectangle(disp, (x1,y1), (x2,y2), (190,190,0), 1)

    for veh in motor_vehs:
        x1,y1,x2,y2 = [int(v) for v in veh["box"]]
        lbl = _CLS_LABEL.get(veh["cls"], "VEH")
        vtype = viol_types.get(veh["id"], "")
        if vtype == "WRONG_WAY":
            color = _COLORS["WRONG_WAY"]
            tag   = f"{lbl}#{veh['id']} WRONG WAY"
            cv2.rectangle(disp, (x1-2,y1-2), (x2+2,y2+2), color, 3)
            cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
            cv2.putText(disp, tag, (x1+1, max(15, y1-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0,0,0), 3)
            cv2.putText(disp, tag, (x1, max(14, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)
        elif vtype == "SIGNAL_JUMP":
            color = _COLORS["SIGNAL_JUMP"]
            tag   = f"{lbl}#{veh['id']} SIGNAL JUMP"
            cv2.rectangle(disp, (x1-2,y1-2), (x2+2,y2+2), color, 3)
            cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
            cv2.putText(disp, tag, (x1+1, max(15, y1-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0,0,0), 3)
            cv2.putText(disp, tag, (x1, max(14, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)
        else:
            color = _COLORS["SCAN"]
            cv2.rectangle(disp, (x1,y1), (x2,y2), color, 1)
            cv2.putText(disp, f"{lbl}#{veh['id']}", (x1, max(14, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    for veh in bikes:
        x1,y1,x2,y2 = [int(v) for v in veh["box"]]
        vid = veh["id"]
        lbl = _CLS_LABEL.get(veh["cls"], "MOTO")
        vtype = viol_types.get(vid, "")
        if vtype == "NO_HELMET":
            color = _COLORS["NO_HELMET"]
            tag   = f"{lbl}#{vid} NO HELMET"
            # Draw thick alert border
            cv2.rectangle(disp, (x1-2,y1-2), (x2+2,y2+2), color, 3)
        elif vtype == "TRIPLE_RIDING":
            color = _COLORS["TRIPLE_RIDING"]
            tag   = f"{lbl}#{vid} TRIPLE!"
            cv2.rectangle(disp, (x1-2,y1-2), (x2+2,y2+2), color, 3)
        elif vtype == "WRONG_WAY":
            color = _COLORS["WRONG_WAY"]
            tag   = f"{lbl}#{vid} WRONG WAY"
            cv2.rectangle(disp, (x1-2,y1-2), (x2+2,y2+2), color, 3)
        elif vtype == "SIGNAL_JUMP":
            color = _COLORS["SIGNAL_JUMP"]
            tag   = f"{lbl}#{vid} SIGNAL JUMP"
            cv2.rectangle(disp, (x1-2,y1-2), (x2+2,y2+2), color, 3)
        else:
            color = _COLORS["OK"]
            tag   = f"{lbl}#{vid}"
        if _HELMET_DEBUG and veh.get("hv"):
            tag += f" H:{veh['hv'][0]}/{veh['hv'][1]}"      # no-helmet votes / helmet votes
        if _TRIPLE_DEBUG:
            tag += f" R:{veh.get('riders', 0)}"
            for rb in veh.get("rider_boxes", []):
                rx1, ry1, rx2, ry2 = [int(v) for v in rb]
                cv2.rectangle(disp, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
        cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
        cv2.putText(disp, tag, (x1+1, max(15, y1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0,0,0), 3)
        cv2.putText(disp, tag, (x1, max(14, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)

    # Stop line + signal box + signal state (drawn on every frame, incl. skipped ones)
    _draw_signal_overlay(disp)
    # Wrong-way zone(s) + allowed-direction arrow
    _draw_wrong_way_overlay(disp)
    return disp


async def _post_to_judge(vehicle_id: int, violation_type: str,
                          crop_bytes: bytes, frame_id: int) -> None:
    """Fire-and-forget POST to the Judge Feed server on port 8001."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(
                JUDGE_SERVER,
                data={"vehicle_id": str(vehicle_id),
                      "violation_type": violation_type,
                      "frame_id": str(frame_id)},
                files={"image": ("crop.jpg", crop_bytes, "image/jpeg")},
            )
    except Exception:
        pass   # judge server offline — ignore silently


def _two_stage_process_frame(
    frame: np.ndarray,
    frame_id: int,
    model,
    confirmed_ids: set,
    known_violations: Optional[dict] = None,
) -> tuple:
    """
    Run the two-stage detection on a single frame.
    Returns (annotated_frame, violations_this_frame, bikes, persons, motor_vehs).

    `confirmed_ids` is a shared set to avoid re-flagging the same vehicle.
    `known_violations` is a shared dict {vehicle_id: violation_type} that persists
    across ALL frames so previously confirmed vehicles keep their color box even
    when they produce no new violation in the current frame.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── COCO tracking ────────────────────────────────────────────────────
    try:
        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=_COCO_CONF, classes=_ALL_CLASSES,
            device=_DEVICE, verbose=False, imgsz=_pick_imgsz(frame), iou=_COCO_IOU,
        )
    except Exception as e:
        print(f"[YOLO ERR] {e}")
        # return 5 values so the caller's unpacking doesn't crash
        return frame.copy(), [], [], [], []

    persons   : list = []
    bikes     : list = []
    motor_vehs: list = []
    violations: list = []   # NEW violations detected in this frame only

    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        res    = results[0]
        bboxes = res.boxes.xyxy.cpu().numpy()
        clses  = res.boxes.cls.cpu().numpy().astype(int)
        confs  = res.boxes.conf.cpu().numpy()
        ids    = (res.boxes.id.cpu().numpy().astype(int)
                  if res.boxes.id is not None
                  else np.full(len(clses), -1, dtype=int))

        for i in range(len(clses)):
            box = bboxes[i].tolist()
            det = {"box": box, "conf": float(confs[i]),
                   "id": int(ids[i]), "cls": int(clses[i]),
                   "bw": box[2]-box[0], "bh": box[3]-box[1]}
            c = det["cls"]
            if   c == _CLS_PERSON:    persons.append(det)
            elif c in _BIKE_CLASSES:  bikes.append(det)
            elif c in _MOTOR_CLASSES: motor_vehs.append(det)

    # FIX: the higher NMS IoU lets crowded riders survive, but can also leave
    # near-identical copies — remove those ourselves.
    persons_all = persons                        # raw detections
    persons_h   = _dedupe_dets(persons_all, 0.85)   # permissive set -> helmet checks (never lose a pillion)
    persons     = _dedupe_persons(persons_all)      # strict set -> triple-riding count
    bikes   = _dedupe_dets(bikes,   _BIKE_DEDUPE_IOU)   # also merges "motorcycle"+"bicycle" on the same bike
    motor_vehs = _dedupe_dets(motor_vehs, _VEH_DEDUPE_IOU)   # FIX: "car"+"truck" on the same car = one vehicle

    # FIX: merge tracker ID switches into one canonical bike id
    _resolve_bike_ids(bikes, frame_id)
    # FIX: same for cars / buses / trucks — their ids used to switch silently and the
    # signal-jump check lost the vehicle's previous position (so the crossing was never seen)
    _resolve_bike_ids(motor_vehs, frame_id, _veh_alias, _veh_track_last)

    # ── Bike violation checks ─────────────────────────────────────────────
    # each person is assigned to ONE bike only (using the extended rider zone)
    rider_map   = _assign_riders(bikes, persons)                 # strict  (triple riding)
    rider_map_h = _assign_riders(bikes, persons_h, loose=True)   # looser  (helmet checks)

    helmet_dets = _helmet_detect_full(frame) if bikes else []
    for idx, veh in enumerate(bikes):
        vid, box = veh["id"], veh["box"]
        bw, bh   = veh["bw"], veh["bh"]

        if bw < _MIN_VEH_PX or bh < _MIN_VEH_PX or vid < 0:
            continue

        riders = rider_map[idx]
        count  = len(riders)
        veh["riders"]      = count                          # shown by TRIPLE_DEBUG overlay
        veh["rider_boxes"] = [r["box"] for r in riders]

        # ── Triple riding ──
        # FIX: count hits in a sliding window instead of "N frames in a row".
        # Riders overlap heavily, so 3 people are not separable on EVERY frame;
        # one frame with only 2 boxes used to reset the counter to zero.
        hist = _triple_hist.setdefault(vid, deque(maxlen=_TRIPLE_WINDOW))
        if count >= 3 and min(bw, bh) >= _TRIPLE_MIN_BIKE_PX:
            hist.append(1)
            _triple_conf[vid] = sum(r["conf"] for r in riders) / count
        else:
            hist.append(0)

        if sum(hist) >= _TRIPLE_MIN_HITS:
            key = (vid, "TRIPLE_RIDING")
            if key not in confirmed_ids:
                confirmed_ids.add(key)
                try:
                    cb = _encode_jpeg(_crop_head(frame, box))
                except Exception:
                    cb = b""
                violations.append({
                    "violation_type": "TRIPLE_RIDING",
                    "vehicle_id": vid,
                    "bbox": box,
                    "confidence": round(_triple_conf.get(vid, 0.0), 3),
                    "frame_id": frame_id,
                    "timestamp": ts,
                    "plate_text": "UNKNOWN",
                    "rider_count": max(count, 3),
                    "_crop_bytes": cb,
                })
        # FIX: no more `continue` here — a triple-riding bike still gets its helmet check.

        # ── Helmet check — confidence-weighted vote over several frames ──
        if (vid, "HELMET_DONE") in confirmed_ids:
            continue
        if _HELMET_MOTO_ONLY and veh["cls"] != _CLS_MOTO:
            continue                  # pedal cyclists don't need a helmet
        verdict, hcrop, hconf = _helmet_verdict_per_rider(frame, box, helmet_dets, rider_map_h[idx])
        if verdict == "SKIP":
            continue                  # unreadable this frame — try again on the next one

        votes = _helmet_votes.setdefault(
            vid, {"no": 0, "yes": 0, "no_w": 0.0, "yes_w": 0.0, "best": None, "best_conf": 0.0})
        if verdict == "NO_HELMET":
            votes["no"] += 1
            votes["no_w"] += hconf
            if hconf >= votes["best_conf"]:
                votes["best"], votes["best_conf"] = hcrop, hconf
        else:
            votes["yes"] += 1
            votes["yes_w"] += hconf

        total = votes["no"] + votes["yes"]
        veh["hv"] = (votes["no"], votes["yes"])          # shown by HELMET_DEBUG overlay
        if total < _HELMET_MIN_VOTES:
            continue

        no_ratio = votes["no_w"] / max(votes["no_w"] + votes["yes_w"], 1e-6)
        no_mean  = votes["no_w"] / max(votes["no"], 1)

        if (votes["no"] >= _HELMET_MIN_NO_VOTES and no_ratio >= _HELMET_NO_RATIO
                and no_mean >= _HELMET_NO_MEAN_CONF):
            confirmed_ids.add((vid, "HELMET_DONE"))
            if _is_duplicate_no_helmet(frame_id, box):
                if _HELMET_DEBUG: print(f"[helmet] f{frame_id} bike#{vid} no-helmet SUPPRESSED (same spot as an earlier alert)")
                continue
            if _LOOK_DEDUPE_ON:
                look = _rider_descriptor(frame, box)
                if look is not None:
                    if _is_same_rider_look(frame_id, look):
                        if _HELMET_DEBUG: print(f"[helmet] f{frame_id} bike#{vid} no-helmet SUPPRESSED (looks like a rider already logged)")
                        continue                  # same-looking rider already logged — don't log twice
                    _recent_no_helmet_look.append((frame_id, look))
            _recent_no_helmet.append((frame_id, box))
            if _HELMET_DEBUG:
                print(f"[helmet] f{frame_id} bike#{vid} *** NO_HELMET "
                      f"(no={votes['no']} yes={votes['yes']} ratio={no_ratio:.2f} mean={no_mean:.2f})")
            try:
                cb = _encode_jpeg(votes["best"])
            except Exception:
                cb = b""
            violations.append({
                "violation_type": "NO_HELMET",
                "vehicle_id": vid,
                "bbox": box,
                "confidence": round(votes["best_conf"], 3),
                "frame_id": frame_id,
                "timestamp": ts,
                "plate_text": "UNKNOWN",
                "rider_count": count,
                "_crop_bytes": cb,
            })
        elif total >= _HELMET_CLEAR_VOTES and votes["yes"] >= _HELMET_CLEAR_VOTES and no_ratio < 0.25:
            # Only repeated, clear helmet detections close the track. A weak/uncertain track
            # must never be silently treated as compliant.
            confirmed_ids.add((vid, "HELMET_DONE"))
            if _HELMET_DEBUG:
                print(f"[helmet] f{frame_id} bike#{vid} HELMET_CONFIRMED (no_ratio={no_ratio:.2f})")
        elif total >= _HELMET_MAX_VOTES:
            # Still uncertain: discard the old window and keep looking at later, clearer frames.
            # This prevents one early run of bad frames from permanently hiding a no-helmet rider.
            votes.update({"no": 0, "yes": 0, "no_w": 0.0, "yes_w": 0.0,
                          "best": None, "best_conf": 0.0})
            if _HELMET_DEBUG:
                print(f"[helmet] f{frame_id} bike#{vid} UNCERTAIN; resetting vote window")

    # ── Signal jump (red light) — bikes AND cars/buses/trucks ──────────────
    # Only runs when at least one signal box AND at least one stop line are configured.
    if _signal_cfg["rois"] and _signal_cfg["lines"]:
        raw   = _signal_raw_reading(frame, _signal_cfg["rois"])
        state = _update_signal_state(raw, frame_id)
        violations += _check_signal_jump(
            bikes + motor_vehs, state, _signal_cfg["red_onset"], _signal_cfg["lines"],
            _signal_cfg["dir"], frame, frame_id, ts, confirmed_ids,
        )

    # ── Wrong way (bikes AND cars/buses/trucks) ──
    # Only runs when at least one zone is configured (WRONG_WAY env or ?wrong_way= query param).
    if _ww_cfg["rules"]:
        violations += _check_wrong_way(
            bikes + motor_vehs, _ww_cfg["rules"], frame, frame_id, ts, confirmed_ids,
        )

    # Draw with persistent known_violations so confirmed boxes stay colored
    annotated = _annotate(frame, bikes, persons, motor_vehs, violations, known_violations)
    # Return detection lists so the caller can cache them for skipped-frame annotation
    return annotated, violations, bikes, persons, motor_vehs


# ─── Pydantic response models ─────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"


class DetectionOut(BaseModel):
    cls: str
    conf: float
    bbox: List[float]
    track_id: int


class ViolationOut(BaseModel):
    violation_type: str
    vehicle_id: int
    bbox: List[float]
    confidence: float
    frame_id: int
    timestamp: str
    plate_text: str
    rider_count: int
    detection_tier: Optional[str] = None
    ocr_confidence_raw: Optional[float] = None
    plate_valid: bool = False


class DetectImageResponse(BaseModel):
    detections: List[DetectionOut]
    violations: List[ViolationOut]
    annotated_image_b64: str = Field(
        description="JPEG annotated frame as base64 — use as <img src='data:image/jpeg;base64,...'>"
    )
    frame_id: int = 0
    timestamp: str
    challans: List[Dict[str, Any]] = []


class ALPRResponse(BaseModel):
    raw_text: str
    plate_number: str
    confidence: float
    detection_tier: Optional[str] = None
    ocr_confidence_raw: Optional[float] = None
    plate_valid: bool = False


class ViolationRecord(BaseModel):
    violation_id: str = ""
    plate_number: str = ""
    violation_type: str = ""
    confidence: float = 0.0
    timestamp: str = ""
    frame_id: Any = 0
    plate_valid: bool = False
    image_path: str = ""


class ViolationLogResponse(BaseModel):
    count: int
    records: List[Dict[str, Any]]


class ViolationSummaryResponse(BaseModel):
    total: int
    NO_HELMET: int
    TRIPLE_RIDING: int
    SIGNAL_JUMP: int
    WRONG_WAY: int = 0


class DatasetFetchRoboflowRequest(BaseModel):
    api_key: str
    workspace: str
    project: str
    version: int
    format: str = "yolov8"


class DatasetFetchResponse(BaseModel):
    status: str
    message: str
    path: str


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode raw bytes to BGR numpy image, raising 400 on failure."""
    arr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image — ensure it's a valid JPEG/PNG/BMP.")
    return frame


def _frame_to_b64(frame: np.ndarray) -> str:
    """Encode a BGR frame to JPEG base64 string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _detection_to_dict(d) -> dict:
    return {
        "cls": d.cls,
        "conf": round(float(d.conf), 4),
        "bbox": [round(float(x), 2) for x in d.bbox],
        "track_id": int(d.track_id),
    }


def _violation_to_dict(v) -> dict:
    return {
        "violation_type": v.violation_type,
        "vehicle_id": int(v.vehicle_id),
        "bbox": [round(float(x), 2) for x in v.bbox],
        "confidence": round(float(v.confidence), 4),
        "frame_id": int(v.frame_id),
        "timestamp": v.timestamp,
        "plate_text": getattr(v, "plate_text", "UNKNOWN"),
        "rider_count": int(v.rider_count),
        "detection_tier": getattr(v, "detection_tier", None),
        "ocr_confidence_raw": getattr(v, "ocr_confidence_raw", None),
        "plate_valid": getattr(v, "plate_valid", False),
    }


def _run_alpr_on_violation(frame: np.ndarray, violation, alpr):
    """Detect plate within a vehicle bbox and return PlateResult or None."""
    try:
        # Support both object and dict formats for bbox
        bbox = getattr(violation, "bbox", None) or violation.get("bbox")
        plate_bboxes, tier = alpr.detect_plates_in_vehicle(frame, bbox)
        if plate_bboxes:
            result = alpr.read_plate_from_frame(frame, plate_bboxes[0])
            if result:
                result.detection_tier = tier
                return result
    except Exception:
        pass
    return None


def _create_draft_challan_for_violation(violation):
    """Store a draft e-challan after ALPR has supplied a readable plate."""
    _plate = (violation.get("plate_text") if isinstance(violation, dict)
              else getattr(violation, "plate_text", None)) or ""
    if str(_plate).strip().upper() in ("", "UNKNOWN"):
        return None
    try:
        from api.challan_store import create_draft_challan
        if isinstance(violation, dict):
            payload = violation
        else:
            payload = {
                "violation_type": getattr(violation, "violation_type", "UNKNOWN"),
                "vehicle_id": getattr(violation, "vehicle_id", None),
                "bbox": getattr(violation, "bbox", []),
                "confidence": getattr(violation, "confidence", 0),
                "frame_id": getattr(violation, "frame_id", 0),
                "timestamp": getattr(violation, "timestamp", datetime.now().isoformat()),
                "plate_text": getattr(violation, "plate_text", "UNKNOWN"),
                "detection_tier": getattr(violation, "detection_tier", None),
                "plate_valid": getattr(violation, "plate_valid", False),
                "ocr_confidence_raw": getattr(violation, "ocr_confidence_raw", None),
            }
        return create_draft_challan(payload)
    except Exception as exc:
        print(f"[challan] Could not save draft challan: {exc}")
        return None


# ─── Endpoints ────────────────────────────────────────────────────────────────

# GET /health
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check."""
    return {"status": "ok"}


@app.get("/challans", tags=["E-Challan"])
async def get_challans(limit: int = Query(100, ge=1, le=500)):
    """List draft e-challans created from recognized number plates."""
    try:
        from api.challan_store import list_challans
        records = list_challans(limit=limit)
        return {"count": len(records), "records": records}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load challans: {exc}")


@app.get("/challans/{challan_id}", tags=["E-Challan"])
async def get_challan_by_id(challan_id: str):
    """Fetch one draft e-challan by its ID."""
    try:
        from api.challan_store import get_challan
        record = get_challan(challan_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load challan: {exc}")
    if record is None:
        raise HTTPException(status_code=404, detail="Challan not found")
    return record



@app.get("/challans/{challan_id}/pdf", tags=["E-Challan"])
async def download_challan_pdf(challan_id: str):
    from io import BytesIO
    from fastapi.responses import StreamingResponse
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from api.challan_store import get_challan

    record = get_challan(challan_id)

    if record is None:
        raise HTTPException(status_code=404, detail="Challan not found")

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Heading
    pdf.setTitle(f"Gridlock Draft Challan {challan_id}")
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(50, height - 60, "Gridlock - Draft E-Challan")

    pdf.setFont("Helvetica", 10)
    pdf.drawString(50, height - 82, "Automated Traffic Violation Record")
    pdf.line(50, height - 95, width - 50, height - 95)

    # Challan details
    fields = [
        ("Challan ID", record.get("challan_id")),
        ("Status", record.get("status")),
        ("Vehicle Plate", record.get("plate_number")),
        ("Violation Type", record.get("violation_type")),
        ("Violation Time", record.get("violation_time")),
        ("Detection Confidence", record.get("confidence")),
        ("Fine Amount", record.get("fine_amount")),
        ("Frame ID", record.get("frame_id")),
        ("Created At", record.get("created_at")),
    ]

    y = height - 130

    for label, value in fields:
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(50, y, f"{label}:")

        pdf.setFont("Helvetica", 11)
        text_value = str(value if value is not None else "N/A")
        # Keep the PDF safe for standard built-in fonts.
        text_value = text_value.encode("latin-1", "replace").decode("latin-1")
        pdf.drawString(190, y, text_value[:85])

        y -= 28

    # Important notice
    y -= 15
    pdf.line(50, y, width - 50, y)
    y -= 30

    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(50, y, "Important Notice")

    y -= 20
    pdf.setFont("Helvetica", 10)
    pdf.drawString(
        50, y,
        "This document is a prototype draft generated by Gridlock."
    )
    y -= 16
    pdf.drawString(
        50, y,
        "It is not an official government-issued challan or payment demand."
    )

    pdf.save()
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{challan_id}.pdf"'
            )
        },
    )


@app.delete("/challans", tags=["E-Challan"])
async def delete_all_challans_route():
    """Delete all stored e-challans."""
    try:
        from api.challan_store import delete_all_challans
        deleted = delete_all_challans()
        return {"status": "cleared", "deleted": deleted}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete challans: {exc}")

# POST /detect/image
@app.post("/detect/image", response_model=DetectImageResponse, tags=["Detection"])
async def detect_image(
    file: UploadFile = File(..., description="Traffic image (JPEG/PNG/BMP)"),
    conf_threshold: float = Query(0.45, ge=0.1, le=0.9, description="YOLO confidence threshold"),
    overlap_threshold: float = Query(0.30, ge=0.1, le=0.8, description="Person-vehicle overlap threshold"),
    triple_threshold: int = Query(3, ge=2, le=5, description="Riders to trigger triple-riding violation"),
    run_alpr: bool = Query(True, description="Run ALPR plate reading on violated vehicles"),
):
    """
    Detect violations in a single uploaded image.
    Returns detections, violations, and annotated image as base64 JPEG.
    """
    raw = await file.read()
    frame = _decode_image(raw)

    # Apply per-request conf threshold without rebuilding the model
    det = get_detector()
    old_conf = det.conf
    det.conf = conf_threshold

    try:
        detections = det.predict(frame)
    finally:
        det.conf = old_conf

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    checker = get_checker(overlap=overlap_threshold, triple=triple_threshold)
    violations = checker.check(detections, frame_id=0, timestamp=ts)

    # ALPR on each violation
    if run_alpr and violations:
        try:
            alpr = get_alpr()
            for v in violations:
                alpr_res = _run_alpr_on_violation(frame, v, alpr)
                if alpr_res:
                    v.plate_text = alpr_res.plate_number
                    v.detection_tier = alpr_res.detection_tier
                    v.ocr_confidence_raw = alpr_res.ocr_confidence_raw
                    v.plate_valid = alpr_res.plate_valid
                else:
                    v.plate_text = "UNKNOWN"
                    v.detection_tier = None
                    v.ocr_confidence_raw = None
                    v.plate_valid = False
        except HTTPException:
            pass  # ALPR not installed — leave plate_text as UNKNOWN

    # Create a draft e-challan only when ALPR returns a readable plate.
    challan_records = []
    for v in violations:
        challan = _create_draft_challan_for_violation(v)
        if challan:
            challan_records.append(challan)

    # Save to CSV log
    try:
        from src.utils import save_violation_to_csv
        for v in violations:
            save_violation_to_csv(v, v.plate_text)
    except Exception:
        pass

    # Annotate frame
    try:
        from src.utils import annotate_frame
        annotated = annotate_frame(frame, detections, violations)
    except Exception:
        annotated = frame

    return {
        "detections": [_detection_to_dict(d) for d in detections],
        "violations": [_violation_to_dict(v) for v in violations],
        "challans": challan_records,
        "annotated_image_b64": _frame_to_b64(annotated),
        "frame_id": 0,
        "timestamp": ts,
    }


# POST /alpr/read
@app.post("/alpr/read", response_model=ALPRResponse, tags=["ALPR"])
async def alpr_read(
    file: UploadFile = File(..., description="Plate crop image"),
):
    """Read an Indian license plate from a cropped plate image."""
    raw = await file.read()
    frame = _decode_image(raw)

    try:
        alpr = get_alpr()
    except HTTPException as e:
        raise e

    try:
        result = alpr.read_plate_from_crop(frame)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    if result is None:
        raise HTTPException(status_code=400, detail="Could not read plate from image.")

    return ALPRResponse(
        raw_text=result.raw_text,
        plate_number=result.plate_number,
        confidence=result.confidence,
        detection_tier=result.detection_tier,
        ocr_confidence_raw=result.ocr_confidence_raw,
        plate_valid=result.plate_valid,
    )


# GET /violations/log
@app.get("/violations/log", response_model=ViolationLogResponse, tags=["Violations"])
async def violations_log(
    limit: int = Query(50, ge=1, le=500, description="Max records to return"),
):
    """Return today's violation log, newest first."""
    try:
        from src.utils import load_violations_log
        df = load_violations_log()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read log: {e}")

    if df.empty:
        return {"count": 0, "records": []}

    # Newest first, limited
    df = df.iloc[::-1].head(limit)
    records = df.fillna("").to_dict(orient="records")
    return {"count": len(records), "records": records}


# GET /violations/summary
@app.get("/violations/summary", response_model=ViolationSummaryResponse, tags=["Violations"])
async def violations_summary():
    """Return violation counts by type for today."""
    try:
        from src.utils import load_violations_log
        df = load_violations_log()
        if df is None or df.empty or "violation_type" not in df.columns:
            return {"total": 0, "NO_HELMET": 0, "TRIPLE_RIDING": 0, "SIGNAL_JUMP": 0, "WRONG_WAY": 0}

        counts = {
            "total": len(df),
            "NO_HELMET": int((df["violation_type"] == "NO_HELMET").sum()),
            "TRIPLE_RIDING": int((df["violation_type"] == "TRIPLE_RIDING").sum()),
            "SIGNAL_JUMP": int((df["violation_type"] == "SIGNAL_JUMP").sum()),
            "WRONG_WAY": int((df["violation_type"] == "WRONG_WAY").sum()),
        }
        return counts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read log: {e}")

# DELETE /violations/log
@app.delete("/violations/log", tags=["Violations"])
async def clear_violations():
    """Delete today's violation log."""
    try:
        from src.utils import clear_violations_log
        clear_violations_log()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not clear log: {e}")


# WS /ws/detect/video  — Two-stage helmet pipeline (mirrors edge_client.py)
@app.websocket("/ws/detect/video")
async def ws_detect_video(websocket: WebSocket):
    """
    Stream video frames for violation detection using the two-stage pipeline:
      1. COCO YOLOv8s tracks vehicles + persons per frame.
      2. For each motorcycle (bicycles only if HELMET_INCLUDE_BICYCLE=true), a specialized
         helmet YOLO model checks each rider's head zone for NO_HELMET violations.
      3. If a signal box + stop line are configured, any tracked vehicle that
         is behind the stop line when the light turns RED and then crosses it
         while still red is flagged SIGNAL_JUMP.
      4. If a wrong-way zone is configured, any tracked vehicle that keeps moving
         against the zone's allowed direction is flagged WRONG_WAY.
      5. Confirmed violations are also POSTed to the Judge Feed (port 8001).

    Query params (all optional):
      frame_skip   process every Nth frame (default 2)
      signal_roi   x1,y1,x2,y2   box around the traffic light  (resized-frame coords; ";" for several)
      stop_line    x1,y1,x2,y2   two points of the stop line   (resized-frame coords; ";" for several)
      stop_dir     any | forward | reverse   which crossing direction counts (default any)
      stop_flip    1 / 0  (old name: 1 = reverse, 0 = forward)
      wrong_way    x1,y1,x2,y2,dx,dy   zone box + allowed direction (";" for several zones)

    Protocol:
      1. Client sends raw video bytes as first binary message.
      2. Server processes every Nth frame (default 2), sends JSON per frame:
           {frame_id, progress, violations, annotated_image_b64, signal_state}
      3. Server sends {"done": true} on completion.
    """
    global _video_lock
    await websocket.accept()
    tmp_path = None
    lock_held = False

    try:
        # FIX: all tracking state is global, so two videos at once corrupted each other.
        # Videos now run one at a time; a second client simply waits its turn.
        if _video_lock is None:
            _video_lock = asyncio.Lock()
        await _video_lock.acquire()
        lock_held = True

        params     = websocket.query_params
        # FIX: default 2 (was 3) — fast bikes move less between processed frames,
        # so the tracker keeps the same ID far more often.
        frame_skip = max(int(params.get("frame_skip", 2)), 1)

        # Signal-jump config: start from the env defaults for EVERY video (a previous video's
        # ROI / line used to leak into this one), then let the query params override them.
        _signal_cfg.clear()
        _signal_cfg.update(_signal_cfg_from_env())
        roi_q  = _parse_boxes(params.get("signal_roi"), 4)
        line_q = _parse_boxes(params.get("stop_line"), 4)
        if roi_q:
            _signal_cfg["rois"] = roi_q
        if line_q:
            _signal_cfg["lines"] = line_q
        dir_q = (params.get("stop_dir") or "").strip().lower()
        if dir_q in ("any", "forward", "reverse"):
            _signal_cfg["dir"] = dir_q
        elif "stop_flip" in params:
            _signal_cfg["dir"] = ("reverse" if params.get("stop_flip", "").lower() in ("1", "true", "yes")
                                  else "forward")
        if _signal_cfg["rois"] and _signal_cfg["lines"]:
            print(f"[api] Signal-jump ON  rois={_signal_cfg['rois']}  lines={_signal_cfg['lines']}  dir={_signal_cfg['dir']}")
        elif _signal_cfg["rois"] or _signal_cfg["lines"]:
            print("[api][WARN] Signal-jump OFF — only one of signal_roi / stop_line is set, BOTH are required")
        else:
            print("[api] Signal-jump OFF (set SIGNAL_ROI + STOP_LINE, or pass signal_roi + stop_line)")

        # Wrong-way zones: the query param overrides the env value, per video
        _ww_cfg["rules"] = (_parse_boxes(params.get("wrong_way"), 6)
                            or _parse_boxes(os.getenv("WRONG_WAY"), 6))
        if _ww_cfg["rules"]:
            print(f"[api] Wrong-way ON  rules={_ww_cfg['rules']}")
        else:
            print("[api] Wrong-way OFF (set WRONG_WAY, or pass wrong_way=x1,y1,x2,y2,dx,dy)")

        # Receive video bytes from browser
        video_bytes = await websocket.receive_bytes()

        # Write to temp file — cv2.VideoCapture needs a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        # Load models (singletons — cached after first call)
        model = _get_coco_model()
        _get_helmet_model()   # pre-warm so first frame isn't slow

        try:
            alpr = get_alpr()
        except Exception:
            alpr = None

        cap          = cv2.VideoCapture(tmp_path)
        total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        cap.release()

        confirmed_ids    : set  = set()   # vehicles already flagged (never re-flag)
        known_violations : dict = {}      # {vehicle_id: violation_type} — persists all video
        # FIX: fresh counters, votes, ID aliases AND a fresh ByteTrack tracker for every video
        _reset_tracking_state(model)
        # Cache the last detected objects so we can annotate skipped frames too
        _last_bikes     : list = []
        _last_persons   : list = []
        _last_motor_vehs: list = []
        frame_id = 0

        cap = cv2.VideoCapture(tmp_path)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1

                # Resize wide frames
                h0, w0 = frame.shape[:2]
                if w0 > 1280:
                    scale = 1280 / w0
                    frame = cv2.resize(frame, (1280, int(h0 * scale)))

                # Skip frames for speed — but STILL annotate with cached detections
                # so bounding boxes don't flicker off between processed frames.
                if frame_id % frame_skip != 0:
                    progress = round(min(frame_id / total_frames, 1.0), 4)
                    annotated_skip = _annotate(
                        frame, _last_bikes, _last_persons, _last_motor_vehs, [], known_violations
                    )
                    payload = {
                        "frame_id": frame_id,
                        "progress": progress,
                        "violations": [],
                        "annotated_image_b64": _frame_to_b64(annotated_skip),
                        "signal_state": _signal_cfg["state"],
                    }
                    await websocket.send_text(json.dumps(payload, default=str))
                    continue

                annotated, violations, _last_bikes, _last_persons, _last_motor_vehs = \
                    await asyncio.get_running_loop().run_in_executor(
                        None, _two_stage_process_frame,
                        frame, frame_id, model, confirmed_ids, known_violations)

                # Bridge confirmed violations to Judge Feed
                for v in violations:
                    crop_bytes = v.pop("_crop_bytes", b"")
                    if crop_bytes:
                        await _post_to_judge(
                            v["vehicle_id"], v["violation_type"],
                            crop_bytes, frame_id
                        )

                    # ALPR and CSV Logging
                    alpr_res = await asyncio.get_running_loop().run_in_executor(
                        None, _run_alpr_on_violation, frame, v, alpr)
                    if alpr_res:
                        v["plate_text"] = alpr_res.plate_number
                        v["detection_tier"] = alpr_res.detection_tier
                        v["ocr_confidence_raw"] = alpr_res.ocr_confidence_raw
                        v["plate_valid"] = alpr_res.plate_valid
                    else:
                        v["plate_text"] = "UNKNOWN"
                        v["detection_tier"] = None
                        v["ocr_confidence_raw"] = None
                        v["plate_valid"] = False

                    # Create a draft e-challan once ALPR has read the number plate.
                    challan = _create_draft_challan_for_violation(v)
                    if challan:
                        v["challan"] = challan

                    try:
                        from src.utils import save_violation_to_csv
                        save_violation_to_csv(v, v["plate_text"])
                    except Exception:
                        pass

                progress = round(min(frame_id / total_frames, 1.0), 4)
                payload  = {
                    "frame_id":           frame_id,
                    "progress":           progress,
                    "violations":         violations,
                    "annotated_image_b64": _frame_to_b64(annotated),
                    "signal_state":       _signal_cfg["state"],
                }
                await websocket.send_text(json.dumps(payload, default=str))

        finally:
            cap.release()

        await websocket.send_text(json.dumps({"done": True}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
    finally:
        if lock_held and _video_lock is not None:
            _video_lock.release()
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink(missing_ok=True)


# POST /dataset/fetch/roboflow
@app.post("/dataset/fetch/roboflow", response_model=DatasetFetchResponse, tags=["Dataset"])
async def dataset_fetch_roboflow(body: DatasetFetchRoboflowRequest):
    """
    Pull a Roboflow dataset version via the roboflow Python package.
    Requires: pip install roboflow
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="roboflow package not installed. Run: pip install roboflow",
        )

    dest = "data/raw/roboflow"
    try:
        rf = Roboflow(api_key=body.api_key)
        project = rf.workspace(body.workspace).project(body.project)
        version = project.version(body.version)
        version.download(body.format, location=dest)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Roboflow API error: {e}",
        )

    return {
        "status": "ok",
        "message": f"Downloaded {body.workspace}/{body.project} v{body.version} ({body.format}) to {dest}",
        "path": str(Path(dest).resolve()),
    }


# POST /dataset/fetch/kaggle
@app.post("/dataset/fetch/kaggle", response_model=DatasetFetchResponse, tags=["Dataset"])
async def dataset_fetch_kaggle(
    dataset_slug: str = Query(..., description="Kaggle dataset slug: owner/name"),
):
    """
    Pull a Kaggle dataset via kagglehub.
    Requires kaggle.json in ~/.kaggle/ and pip install kagglehub.
    """
    try:
        import kagglehub
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="kagglehub not installed. Run: pip install kagglehub",
        )

    try:
        path = kagglehub.dataset_download(dataset_slug)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Kaggle API error: {e}",
        )

    return {
        "status": "ok",
        "message": f"Downloaded {dataset_slug} via kagglehub",
        "path": str(path),
    }


if __name__ == "__main__":
    # `python -m api.main` also works, with the websocket size limit already raised
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, ws_max_size=1024 * 1024 * 1024)

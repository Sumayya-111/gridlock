"""
Usage:  python apply_fixes.py path/to/api/main.py
Makes a backup (main.py.bak), applies every fix, and STOPS with a clear message
if any expected text is not found exactly once (so nothing is half-applied).
"""
import sys, shutil

NEW_HELMET_FN = '''def _helmet_verdict_per_rider(frame: np.ndarray, bike_box: list, dets: list, riders: list = None):
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


def _is_duplicate_no_helmet(frame_id: int, box: list) -> bool:'''

# (name, old, new, expected_count)
EDITS = [
 ("usage line: bigger websocket limit",
  "  uvicorn api.main:app --reload --port 8000",
  "  uvicorn api.main:app --reload --port 8000 --ws-max-size 1073741824   (1 GB; default 16 MB breaks video upload)", 1),

 ("helmet thresholds (fewer false fines)",
  "_HELMET_MIN_VOTES    = 2      # fewer readable frames needed for distant/brief riders\n"
  "_HELMET_MIN_NO_VOTES = 1      # two independent readings plus weighted confidence\n"
  "_HELMET_NO_RATIO     = 0.50   # weighted majority of no-helmet evidence",
  "_HELMET_MIN_VOTES    = 3      # readable frames needed before judging\n"
  "_HELMET_MIN_NO_VOTES = 2      # at least 2 separate no-helmet readings\n"
  "_HELMET_NO_RATIO     = 0.60   # weighted majority of no-helmet evidence", 1),

 ("per-rider helmet function", "def _is_duplicate_no_helmet(frame_id: int, box: list) -> bool:", NEW_HELMET_FN, 1),

 ("use per-rider helmet function",
  "verdict, hcrop, hconf = _helmet_verdict_for_bike(frame, box, helmet_dets, rider_map_h[idx])",
  "verdict, hcrop, hconf = _helmet_verdict_per_rider(frame, box, helmet_dets, rider_map_h[idx])", 1),

 ("challans field in /detect/image response",
  "    frame_id: int = 0\n    timestamp: str\n",
  "    frame_id: int = 0\n    timestamp: str\n    challans: List[Dict[str, Any]] = []\n", 1),

 ("no challan without a plate",
  '    """Store a draft e-challan after ALPR has supplied a readable plate."""\n    try:',
  '    """Store a draft e-challan after ALPR has supplied a readable plate."""\n'
  '    _plate = (violation.get("plate_text") if isinstance(violation, dict)\n'
  '              else getattr(violation, "plate_text", None)) or ""\n'
  '    if str(_plate).strip().upper() in ("", "UNKNOWN"):\n'
  '        return None\n'
  '    try:', 1),

 ("summary on empty log",
  "        df = load_violations_log()\n\n        counts = {",
  "        df = load_violations_log()\n"
  "        if df is None or df.empty or \"violation_type\" not in df.columns:\n"
  "            return {\"total\": 0, \"NO_HELMET\": 0, \"TRIPLE_RIDING\": 0, \"SIGNAL_JUMP\": 0, \"WRONG_WAY\": 0}\n\n"
  "        counts = {", 1),

 ("json crash on datetime/numpy", "json.dumps(payload)", "json.dumps(payload, default=str)", 2),

 ("remove unicode warning sign (cv2 shows ?)", 'NO HELMET ⚠"', 'NO HELMET"', 1),
 ("remove unicode warning sign 2", 'TRIPLE! ⚠"', 'TRIPLE!"', 1),

 ("wrong-way: track last-seen", '_ww_streak : dict = {}   # {vehicle_id: consecutive "wrong way" readings}',
  '_ww_streak : dict = {}   # {vehicle_id: consecutive "wrong way" readings}\n_ww_seen   : dict = {}   # {vehicle_id: last frame seen}', 1),
 ("wrong-way: reset last-seen", "    _ww_streak.clear()\n", "    _ww_streak.clear()\n    _ww_seen.clear()\n", 1),
 ("wrong-way: record last-seen", '        hist.append(foot)\n        if (vid, "WRONG_WAY") in confirmed_ids',
  '        hist.append(foot)\n        _ww_seen[vid] = frame_id\n        if (vid, "WRONG_WAY") in confirmed_ids', 1),
 ("wrong-way: smarter cleanup",
  "    if len(_ww_hist) > 400:\n        for k in list(_ww_hist)[:200]:\n            _ww_hist.pop(k, None)\n            _ww_streak.pop(k, None)",
  "    if len(_ww_hist) > 400:\n        for k in [k for k, f in _ww_seen.items() if frame_id - f > 900]:\n"
  "            _ww_hist.pop(k, None)\n            _ww_streak.pop(k, None)\n            _ww_seen.pop(k, None)", 1),

 ("PDF branding", "TrafficIQ", "Gridlock", 3),

 ("run YOLO in a thread (server no longer freezes)",
  "                annotated, violations, _last_bikes, _last_persons, _last_motor_vehs = \\\n"
  "                    _two_stage_process_frame(\n"
  "                        frame, frame_id, model, confirmed_ids, known_violations\n"
  "                    )",
  "                annotated, violations, _last_bikes, _last_persons, _last_motor_vehs = \\\n"
  "                    await asyncio.get_running_loop().run_in_executor(\n"
  "                        None, _two_stage_process_frame,\n"
  "                        frame, frame_id, model, confirmed_ids, known_violations)", 1),

 ("run ALPR in a thread",
  "                    # ALPR and CSV Logging\n                    alpr_res = _run_alpr_on_violation(frame, v, alpr)",
  "                    # ALPR and CSV Logging\n"
  "                    alpr_res = await asyncio.get_running_loop().run_in_executor(\n"
  "                        None, _run_alpr_on_violation, frame, v, alpr)", 1),
]

APPEND = '''

if __name__ == "__main__":
    # `python -m api.main` also works, with the websocket size limit already raised
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, ws_max_size=1024 * 1024 * 1024)
'''

def apply(src: str):
    for name, old, new, n in EDITS:
        c = src.count(old)
        if c != n:
            sys.exit(f"STOPPED at '{name}': expected {n} match(es), found {c}. Nothing was written.")
        src = src.replace(old, new)
    if "ws_max_size=1024" not in src:
        src = src.rstrip("\n") + "\n" + APPEND
    return src

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python apply_fixes.py path/to/api/main.py")
    path = sys.argv[1]
    src = open(path, encoding="utf-8").read()
    out = apply(src)
    shutil.copy(path, path + ".bak")
    open(path, "w", encoding="utf-8").write(out)
    import py_compile; py_compile.compile(path, doraise=True)
    print(f"Applied {len(EDITS)} fixes. Backup: {path}.bak. Syntax OK.")
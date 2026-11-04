import cv2, sys

path = sys.argv[1]
frame_no = int(sys.argv[2]) if len(sys.argv) > 2 else 0
cap = cv2.VideoCapture(path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
ok, frame = cap.read()
if not ok:
    sys.exit("Could not read that frame")

# same resize the API does, so coordinates match
h0, w0 = frame.shape[:2]
if w0 > 1280:
    frame = cv2.resize(frame, (1280, int(h0 * 1280 / w0)))
H, W = frame.shape[:2]
vs = min(1.0, 900 / H, 1400 / W)          # shrink only for display
pts = []
PROMPTS = ["1/4 click TOP-LEFT of the signal lamp", "2/4 click BOTTOM-RIGHT of the signal lamp",
           "3/4 click LEFT end of the stop line", "4/4 click RIGHT end of the stop line"]

def redraw():
    img = cv2.resize(frame, None, fx=vs, fy=vs)
    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (int(x * vs), int(y * vs)), 4, (0, 0, 255), -1)
    if len(pts) >= 2:
        cv2.rectangle(img, (int(pts[0][0]*vs), int(pts[0][1]*vs)), (int(pts[1][0]*vs), int(pts[1][1]*vs)), (0, 255, 255), 2)
    if len(pts) == 4:
        cv2.line(img, (int(pts[2][0]*vs), int(pts[2][1]*vs)), (int(pts[3][0]*vs), int(pts[3][1]*vs)), (255, 0, 255), 2)
    msg = PROMPTS[len(pts)] if len(pts) < 4 else "Done - press any key"
    cv2.putText(img, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(img, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    cv2.imshow("pick", img)

def on_click(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append((int(x / vs), int(y / vs)))
        redraw()

cv2.namedWindow("pick")
cv2.setMouseCallback("pick", on_click)
redraw()
while len(pts) < 4:
    if cv2.waitKey(50) == 27:
        sys.exit()
redraw()
cv2.waitKey(0)

(ax, ay), (bx, by), (cx, cy), (dx, dy) = pts
print("\nPaste these in PowerShell:\n")
print(f'$env:SIGNAL_ROI="{min(ax,bx)},{min(ay,by)},{max(ax,bx)},{max(ay,by)}"')
print(f'$env:STOP_LINE="{cx},{cy},{dx},{dy}"')
print('$env:SIGNAL_DEBUG="true"')
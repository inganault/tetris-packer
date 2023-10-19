#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument('dir', help="working directory")
parser.add_argument('input', help="input video file (eg. mp4)")
parser.add_argument('-f','--from', type=int, default=0, help="start from frame #n")
parser.add_argument('-h','--height', type=int, default=36, help="output height")
parser.add_argument('-w','--width', type=int, default=48, help="output wifth")
parser.add_argument('-p','--preview', action='store_true', help="preview output")
parser.add_argument('-a','--adaptive', action='store_true', help="use adaptive thresholding")
args = parser.parse_args()

output_dir = Path(args.dir)
resolution = (args.width, args.height)
scale = None

vid = cv2.VideoCapture(args.input)
if not vid.isOpened():
    raise Exception('cannot open video')
if getattr(args, 'from') != 0:
    count = getattr(args, 'from')
    vid.set(cv2.CAP_PROP_POS_FRAMES, count)
else:
    count = 0
while vid.isOpened():
    output_dir.mkdir(parents=True, exist_ok=True)
    ret, frame = vid.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, resolution)
    if args.adaptive:
        gray = gray.astype(np.single) / 255
        blur = (cv2.GaussianBlur(gray,(13,13),0) - 0.5) * 0.9 + 0.5
        gray = (gray > blur).astype(np.uint8) * 255
    else:
        gray = (gray > 127).astype(np.uint8) * 255

    np.savez_compressed(output_dir / f'{count:04}.npz', map=gray)
    count += 1

    if args.preview:
        if scale is None:
            scale = frame.shape[0] // args.width
        prev = cv2.resize(gray, (resolution[0] * scale, resolution[1] * scale), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('frame', prev)
        if cv2.waitKey(33) == ord('q'):
            break
vid.release()
cv2.destroyAllWindows()

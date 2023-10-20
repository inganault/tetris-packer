#!/usr/bin/env python3

import colorsys
import cv2
import numpy as np
import itertools
from pathlib import Path
import argparse
import sys

TETROMINO = [
    # pairs of (y offset, x offset)
    [],  # 0 is reserved
    [  # 1 = O
        (0, 0), (0, 1), (1, 0), (1, 1),
    ],
    [  # 2 = I
        (0, 0), (0, 1), (0, 2), (0, 3),
    ],
    [  # 3 = I
        (0, 0), (1, 0), (2, 0), (3, 0),
    ],

    [  # 4 = J
        (0, 1), (1, 1), (2, 1), (2, 0),
    ],
    [  # 5 = J
        (0, 0), (0, 1), (0, 2), (1, 2),
    ],
    [  # 6 = J
        (0, 0), (0, 1), (1, 0), (2, 0),
    ],
    [  # 7 = J
        (0, 0), (1, 0), (1, 1), (1, 2),
    ],

    [  # 8 = L
        (0, 0), (0, 1), (1, 1), (2, 1),
    ],
    [  # 9 = L
        (1, 0), (1, 1), (1, 2), (0, 2),
    ],
    [  # 10 = L
        (0, 0), (1, 0), (2, 0), (2, 1),
    ],
    [  # 11 = L
        (0, 0), (0, 1), (0, 2), (1, 0),
    ],

    [  # 12 = S
        (0, 1), (0, 2), (1, 0), (1, 1),
    ],
    [  # 13 = S
        (0, 0), (1, 0), (1, 1), (2, 1),
    ],
    [  # 14 = Z
        (0, 0), (0, 1), (1, 1), (1, 2),
    ],
    [  # 15 = Z
        (0, 1), (1, 1), (1, 0), (2, 0),
    ],

    [  # 16 = T
        (0, 0), (0, 1), (0, 2), (1, 1),
    ],
    [  # 17 = T
        (0, 0), (1, 0), (2, 0), (1, 1),
    ],
    [  # 18 = T
        (0, 1), (1, 0), (1, 1), (1, 2),
    ],
    [  # 19 = T
        (1, 0), (0, 1), (1, 1), (2, 1),
    ],
]

def status(*args,**kwargs):
    return print(*args, **kwargs, file=sys.stderr)

def hsv2bgr(h, s, v):
    return tuple(
        255 * i for i in colorsys.hsv_to_rgb(h / 360, s, v)[::-1]
        )

TETROMINO_COLOR = [
    # rgb
    hsv2bgr(0, 0.00, 0.70),  # empty
    hsv2bgr(182, 0.20, 0.97),   # 1 = O
    hsv2bgr(59, 0.20, 0.97),    # 2 = I
    hsv2bgr(59, 0.20, 0.97),    # 3 = I
    hsv2bgr(231, 0.20, 0.97),   # 4 = J
    hsv2bgr(231, 0.20, 0.97),   # 5 = J
    hsv2bgr(231, 0.20, 0.97),   # 6 = J
    hsv2bgr(231, 0.20, 0.97),   # 7 = J
    hsv2bgr(34, 0.20, 0.97),    # 8 = L
    hsv2bgr(34, 0.20, 0.97),    # 9 = L
    hsv2bgr(34, 0.20, 0.97),    # 10 = L
    hsv2bgr(34, 0.20, 0.97),    # 11 = L
    hsv2bgr(113, 0.20, 0.97),   # 12 = S
    hsv2bgr(113, 0.20, 0.97),   # 13 = S
    hsv2bgr(0, 0.20, 0.97),     # 14 = Z
    hsv2bgr(0, 0.20, 0.97),     # 15 = Z
    hsv2bgr(302, 0.20, 0.97),   # 16 = T
    hsv2bgr(302, 0.20, 0.97),   # 17 = T
    hsv2bgr(302, 0.20, 0.97),   # 18 = T
    hsv2bgr(302, 0.20, 0.97),   # 19 = T
]
STROKE_COLOR = (80, 80, 80)


parser = argparse.ArgumentParser()
parser.add_argument('dir', help="working directory")
parser.add_argument('-f', '--from', type=int, default=0, help="start from frame #n")
parser.add_argument('-s', '--scale', type=int, default=24, help="block size in pixel")
parser.add_argument('-w', '--width', type=int, default=2, help="stroke width in pixel")
parser.add_argument('-u', '--unfilled', action='store_true', help="paint unfilled blocks")
parser.add_argument('-i', '--info', action='store_true', help="show frame count")
parser.add_argument('-o', '--output', action='store_true', help="output rawvideo to stdout")
args = parser.parse_args()

output_dir = Path(args.dir)
upscale = args.scale
swidth = args.width
single_step_mode = False
paint_unfilled = args.unfilled
resolution_displayed = False


for count in itertools.count(getattr(args, 'from')):
    src = output_dir / f'{count:04d}.npz'
    if not src.exists():
        break
    if args.info:
        status(f"Frame #{count}")
    if paint_unfilled:
        try:
            map = np.load(src)
        except FileNotFoundError:
            status(f'file {src} missing')
            continue
        map = map['map']

    src = output_dir / f'{count:04d}_out.npz'
    try:
        piece = np.load(src)
    except FileNotFoundError:
        status(f'file {src} missing')
        continue
    piece = piece['piece']
    h, w = piece.shape
    frame = np.zeros((h * upscale, w * upscale, 3,), np.uint8)

    if paint_unfilled:
        for y in range(map.shape[0]):
            for x in range(map.shape[1]):
                if map[y, x]:
                    frame[y * upscale:(y + 1) * upscale,
                          x * upscale:(x + 1) * upscale, :] = TETROMINO_COLOR[0]

    for y in range(piece.shape[0]):
        for x in range(piece.shape[1]):
            v = piece[y, x]
            if not v:
                continue
            blocks = TETROMINO[v]
            for (dy, dx) in blocks:
                ny, nx = y + dy, x + dx
                y1, y2 = ny * upscale, (ny + 1) * upscale
                x1, x2 = nx * upscale, (nx + 1) * upscale
                color = TETROMINO_COLOR[v]
                frame[y1 + swidth:y2 - swidth,
                      x1 + swidth:x2 - swidth, :] = color

                frame[y2 - swidth: y2,
                      x1 + swidth:x2 - swidth, :] = color if (dy + 1, dx) in blocks else STROKE_COLOR
                frame[y1: y1 + swidth,
                      x1 + swidth:x2 - swidth, :] = color if (dy - 1, dx) in blocks else STROKE_COLOR
                frame[y1 + swidth:y2 - swidth,
                      x2 - swidth:x2, :] = color if (dy, dx + 1) in blocks else STROKE_COLOR
                frame[y1 + swidth:y2 - swidth,
                      x1:x1 + swidth, :] = color if (dy, dx - 1) in blocks else STROKE_COLOR

                frame[y1: y1 + swidth,
                      x1: x1 + swidth, :] = color if (dy - 1, dx - 1) in blocks and (dy - 1, dx) in blocks and (dy, dx - 1) in blocks else STROKE_COLOR
                frame[y1: y1 + swidth,
                      x2 - swidth:x2, :] = color if (dy - 1, dx + 1) in blocks and (dy - 1, dx) in blocks and (dy, dx + 1) in blocks else STROKE_COLOR
                frame[y2 - swidth: y2,
                      x1: x1 + swidth, :] = color if (dy + 1, dx - 1) in blocks and (dy + 1, dx) in blocks and (dy, dx - 1) in blocks else STROKE_COLOR
                frame[y2 - swidth: y2,
                      x2 - swidth:x2, :] = color if (dy + 1, dx + 1) in blocks and (dy + 1, dx) in blocks and (dy, dx + 1) in blocks else STROKE_COLOR

    if args.info:
        frame = cv2.putText(
            frame, f'{count}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (127, 127, 127), 10)

    if args.output:
        if not resolution_displayed:
            resolution_displayed = True
            status('## Output option:')
            status(f'## ffmpeg -video_size {frame.shape[1]}x{frame.shape[0]} -pix_fmt bgr24 -f rawvideo -i - out.mp4')
        sys.stdout.buffer.write(memoryview(frame))
        sys.stdout.buffer.flush()
        continue

    cv2.imshow('frame', frame)
    if single_step_mode:
        key = cv2.waitKey()
    else:
        key = cv2.waitKey(33)
    if key == ord('q'):
        break
    elif key == ord('e'):
        single_step_mode = True
    elif key == ord(' '):
        single_step_mode = not single_step_mode
cv2.destroyAllWindows()

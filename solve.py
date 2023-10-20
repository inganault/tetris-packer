#!/usr/bin/env python3

import platform
import itertools
from pathlib import Path
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dir', help="working directory")
parser.add_argument('-f', '--from', type=int, default=0, help="start from frame #n")
args = parser.parse_args()

pwd = Path.cwd()
output_dir = Path(args.dir)
project_root = Path(__file__).parent

def run(cmd):
    cmd = [str(i.resolve().relative_to(pwd))
           if isinstance(i, Path) else i for i in cmd]
    print(cmd)
    subprocess.check_call(cmd)

run(['cargo', 'build', '--release'])
if platform.system() == 'Windows':
    exe = project_root / 'target/release/tetris.exe'
else:
    exe = project_root / 'target/release/tetris'
if not exe.exists():
    raise Exception(f'Build failed?, {exe} not found')

for i in itertools.count(getattr(args, 'from')):
    src = output_dir / f'{i:04d}.npz'
    if not src.exists():
        print(f'File {src} not found, stopping')
        break
    out = output_dir / f'{i:04d}_out.npz'
    if not out.exists():
        try:
            if i == 0:
                run([exe, src, '-o', out])
            else:
                ref = output_dir / f'{i-1:04d}_out.npz'
                run([exe, src, ref, '-o', out])
        except KeyboardInterrupt:
            if out.exists():
                out.unlink()
            print(f"Stopped at frame #{i:04}")
            break

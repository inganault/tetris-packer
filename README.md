# Tetris packer

Places tetris pieces (tetromino) into a problem grid using space state search
with crystal-growth-like heuristic.  

It also turns cpu into space heater.

```
from

.##.
####
##.#
.###

into

     ┌────────┐
     │00000000│
     │000┌────┘
┌────┘000│┌────────┐
│00000000││11111111│
└────────┘└────┐111│
┌────────┐     │111│
│22222222│     │111│
└────┐222│     │111│
     │222└────┐│111│
     │22222222││111│
     └────────┘└───┘
```

## Usage

Requires
- Rust
- Python 3 (opencv, numpy)

```sh
# single
python example.py

# Use on video
preprocess.py frames/ input.mp4 -p
solve.py frames/
render.py frames/ -i
```

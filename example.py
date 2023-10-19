import numpy as np
import os
problem = np.array([
    [0,1,1,0],
    [1,1,1,1],
    [1,1,0,1],
    [0,1,1,1],
]).astype(np.uint8)
np.savez_compressed('tmp.npz', map=problem)
os.system('cargo run -- tmp.npz -o tmp_out.npz')
out = np.load('tmp_out.npz')['piece']
print(out)

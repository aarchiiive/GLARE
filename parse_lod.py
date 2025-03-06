import os
from pathlib import Path


data_dir = Path('LOD')
save_dir = Path('LOD/val')
# save_dir.mkdir(exist_ok=True, parents=True)
(save_dir / 'normal').mkdir(exist_ok=True, parents=True)
(save_dir / 'dark').mkdir(exist_ok=True, parents=True)

meta_file = data_dir / 'val.txt'

with open(meta_file, 'r') as f:
    meta_lines = f.readlines()

for line in meta_lines:
    line = line.strip().split()
    filename = Path(line[0]).name
    image_path = data_dir / 'RGB_normal' / filename
    os.symlink(image_path.resolve(), save_dir / 'normal' / filename)
    image_path = data_dir / 'RGB_Dark' / filename
    os.symlink(image_path.resolve(), save_dir / 'dark' / filename)


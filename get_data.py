import os
import glob
import shutil

# TODO: command line args for these?

if not os.path.exists("sem-0.1.0/"):
    os.system("curl -O https://pmb.let.rug.nl/releases/sem-0.1.0.zip")
    os.system("unzip sem-0.1.0.zip")

train_split = 0.7
val_split = 0.15

for subset in ['gold', 'silver']:

    files = glob.glob(f'sem-0.1.0/data/{subset}/*')

    train_dir = f'sem-0.1.0/data/{subset}/train/'
    val_dir = f'sem-0.1.0/data/{subset}/val/'
    test_dir = f'sem-0.1.0/data/{subset}/test/'

    for dirname in [train_dir, val_dir, test_dir]:
        if not os.path.exists(dirname):
            os.mkdir(dirname)

    train_idx = int(train_split * len(files))
    val_size = int(val_split * len(files))

    for filename in files[:train_idx]:
        shutil.move(filename, train_dir)

    for filename in files[train_idx:train_idx+val_size]:
        shutil.move(filename, val_dir)

    for filename in files[train_idx+val_size:]:
        shutil.move(filename, test_dir)

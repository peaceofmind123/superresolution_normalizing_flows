import imageio
import pickle
from tqdm import tqdm
import sys
import os
from prepare_data import get_img_paths, imread
import gc
import numpy
import sPickle

# To run this with default directory structure, use the following command
# python prepare_celeba.py /Users/ashishthesatan/Projects/MSC\ Project/HD-CelebA-Cropper-master/data/aligned/align/hr /Users/ashishthesatan/Projects/MSC\ Project/HD-CelebA-Cropper-master/data/aligned/align/celebA-hr.pklv4
def read_imgs_generator(img_dir):
    img_paths = get_img_paths(img_dir)
    for i, path in tqdm(enumerate(img_paths)):
        img = imread(path)
        yield img


def main(img_dir, pkl_path):
    generator = read_imgs_generator(img_dir)
    with open(pkl_path, 'wb') as f:
        sPickle.s_dump(generator, f)


if __name__ == "__main__":
    img_dir = sys.argv[1]
    pkl_path = sys.argv[2]
    assert os.path.isdir(img_dir)
    
    main(img_dir, pkl_path)        


def to_pklv4(obj, path, vebose=False):
    create_all_dirs(path)
    with open(path, 'wb') as f:
        dump(obj, f, protocol=4)
    if vebose:
        print("Wrote {}".format(path))


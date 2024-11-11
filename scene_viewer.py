import pyrender
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary
import numpy as np
import os
from argparse import ArgumentParser
import json



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    split_file = os.path.join(args.model, "split.json")
    if os.path.exists(split_file):
        train_test_split = json.load(open(split_file))
    # intrinsics = read_intrinsics_binary(os.path.join(args.model, "sparse/cameras.bin"))
    # extrinsics = read_extrinsics_binary(os.path.join(args.model, "sparse/images.bin"))
    # print(extrinsics)
    
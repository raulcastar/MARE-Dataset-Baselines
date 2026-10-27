"""
Unpacks the Hugging Face download of the MARE Lunar Rover Dataset into the
folder layout the baselines read:

    python -m mare_baselines.unpack_hf <download folder> <dataset folder>

<download folder>: the repository as downloaded (`hf download
raulcastar/mare-lunar-rover-dataset --repo-type dataset --local-dir ...`), with
data/<split>/*.tar and episodes/. Writes <dataset folder>/samples/<episode>/
<field>/<frame>.<ext> and samples/<episode>/episode.json.
"""
import argparse
import glob
import os
import shutil
import tarfile


def unpack(src, dst):
    for tar_path in sorted(glob.glob(os.path.join(src, "data", "*", "*.tar"))):
        with tarfile.open(tar_path) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                key, field = member.name.split(".", 1)          # <episode>_<frame>.<field>.<ext>
                if field == "json":
                    continue                                    # the same metadata as frames.csv
                episode, frame = key.rsplit("_", 1)
                folder, ext = field.rsplit(".", 1)
                out = os.path.join(dst, "samples", episode, folder)
                os.makedirs(out, exist_ok=True)
                with tar.extractfile(member) as f, open(os.path.join(out, f"{frame}.{ext}"), "wb") as g:
                    shutil.copyfileobj(f, g)
        print(os.path.basename(os.path.dirname(tar_path)), os.path.basename(tar_path), flush=True)
    for meta in glob.glob(os.path.join(src, "episodes", "*.json")):
        episode = os.path.basename(meta)[:-5]
        os.makedirs(os.path.join(dst, "samples", episode), exist_ok=True)
        shutil.copy(meta, os.path.join(dst, "samples", episode, "episode.json"))
        frames = os.path.join(src, "episodes", f"{episode}_frames.csv")
        if os.path.isfile(frames):
            shutil.copy(frames, os.path.join(dst, "samples", episode, "frames.csv"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()
    unpack(args.src, args.dst)


if __name__ == "__main__":
    main()

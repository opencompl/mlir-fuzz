import os
import shutil


def remove_directories_with_one_file(path: str):
    for root, dirs, files in os.walk(path):
        if len(files) == 1 and len(dirs) == 0:
            print(f"Removing {root} because it only contains one file")
            shutil.rmtree(root)


if __name__ == "__main__":
    remove_directories_with_one_file("results")

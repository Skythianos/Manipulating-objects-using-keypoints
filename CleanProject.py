import os
import shutil

out_path = "out"
dirs = os.listdir(out_path)
for dir in dirs:
    curdir = os.path.join(out_path, dir)
    if not os.path.isdir(curdir):
        continue

    contents = os.listdir(curdir)
    if len(contents) < 10 and '_' in curdir:
        shutil.rmtree(curdir)
        print curdir

# print [[f for f in dirdata[2] if f.endswith(".bak")] for dirdata in os.walk("arrangement_calib")]
for dirdata in os.walk("."):
    for f in dirdata[2]:
        if f.endswith(".bak"):
            path = os.path.join(dirdata[0], f)
            print path
            os.remove(path)
import cv2
import numpy as np
import cPickle as pickle
from os.path import isfile
from glob import glob

FEATURE_SURF = "surf"

def getDetector(name):
    name = name.lower()
    if name == "surf":
        return cv2.SURF()

    return cv2.SURF()

# dump file name: "cache/filename.detectorType.p"
class FeatureLoader:
    def __init__(self):
        pass

    def getFileName(self, filename, dType):
        if filename.startswith("out"):
            filename = filename.replace("\\", "/").replace("/", "_")
            return "cache/feature_%s.%s.p" % (filename, dType)
        else:
            filename = filename[filename.rindex("/") + 1:]
            return "cache/feature_%s.%s.p" % (filename, dType)

    """
    :returns kp, des
    """
    def loadFeatures(self, filename, detectorType = "surf"):
        detectorType = detectorType.lower()
        detector = getDetector(detectorType)
        fname = self.getFileName(filename, detectorType)
        if isfile(fname):
            f = open(fname, "rb")
            keyPts = pickle.load(f)
            f.close()
            return self.deserializeKeyPoints(keyPts)

        img = cv2.imread(filename, 0)
        kp, des = detector.detectAndCompute(img, None)
        if des is None:
            des = []

        f = open(fname, "wb")
        pickle.dump(self.serializeKeyPoints(kp, des), f, 2)
        f.close()

        return kp, des

    def serializeKeyPoints(self, kpts, dstors):
        return [
            (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id, des)
            for kp, des in zip(kpts, dstors)]

    def deserializeKeyPoints(self, serd):
        kpts = [cv2.KeyPoint(x=ser[0][0],y=ser[0][1],_size=ser[1], _angle=ser[2],
                            _response=ser[3], _octave=ser[4], _class_id=ser[5]) for ser in serd]
        dstors = [ser[6] for ser in serd]

        return kpts, dstors

def print_rand(arr, idxs):
    sel = [arr[idx] for idx in idxs]
    print(sel)

def drawKpts(img, kpts):
    for kp in kpts:
        cv2.circle(img, tuple(map(int, kp.pt)), 10, (0, 0, 255), 1)
    cv2.imshow("asd", img)
    cv2.waitKey()

if __name__ == "__main__":
    import random
    fl = FeatureLoader()

    files = glob("out/2017_3_8__14_51_22/*.jpg")
    for f in files:
        print  f
        kp, des = fl.loadFeatures(f, "surf")
        # drawKpts(cv2.imread(f), kp)


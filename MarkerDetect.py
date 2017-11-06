import cv2
import numpy as np
import pickle
from Utils import *

click_pos = None
def click(event,x,y,flags,param):
    global click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pos = (x, y)

def update_points(points, newpoint, dist = 100):
    for i in range(len(points)):
        p = points[i]
        if p is None:
            points[i] = newpoint
            return

        d = (p[0] - newpoint[0]) ** 2 + (p[1] - newpoint[1]) ** 2
        if d < dist ** 2:
            points[i] = newpoint
            return

def draw_points(img, points, scale):
    newimg = np.copy(img)
    for p in points:
        if p is not None:
            x, y = (int(p[0] / scale), int(p[1] / scale))
            r = 10
            t = 1
            cv2.circle(newimg, (x, y) , r, (0, 0, 255), t)
            cv2.line(newimg, (x - r, y), (x + r, y), (0, 0, 255), t)
            cv2.line(newimg, (x, y - r), (x, y + r), (0, 0, 255), t)
    return newimg

def detectMarker(filename):
    global click_pos

    points = [None] * 4
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", click)
    img = cv2.imread(filename)
    img_big = cv2.pyrUp(img)
    img_small = cv2.pyrDown(img)
    img_small = cv2.pyrDown(img_small)
    img_small_orig = np.copy(img_small)

    big_width = 50
    big_height = 50
    while True:
        cv2.imshow("img", img_small)
        while click_pos is None:
            c = cv2.waitKey(1)
            if c == 27:
                exit()
            elif c == 13:
                return points

        x_o = 4 * click_pos[0]
        y_o = 4 * click_pos[1]
        click_pos = None

        x = np.max(2 * x_o - big_width, 0)
        y = np.max(2 * y_o - big_height, 0)

        dx = 2 * big_width
        dy = 2 * big_height

        detail = np.array(img_big[y : y + dy, x : x + dx, :])
        cv2.imshow("img2", detail)
        cv2.setMouseCallback("img2", click)
        while click_pos is None:
            c = cv2.waitKey(1)
            if c == 27:
                exit()
            elif c == 13:
                return points

        xp = (x + click_pos[0]) / 2.0 + 5
        yp = (y + click_pos[1]) / 2.0
        click_pos = None

        update_points(points, (xp, yp))
        print(points)
        img_small = draw_points(img_small_orig, points, 4)

def getParams(filename):
    points = detectMarker(filename)
    for p in points:
        if p is None:
            return

    imgPt = np.array([list(p) for p in points], np.float32)

    flag = cv2.CV_ITERATIVE
    retval, rvec, tvec = cv2.solvePnP(objPtMarker.T, imgPt, camMtx, None, flags=flag)

    if retval:
        print(retval, rvec, tvec)
        rmat, jac = cv2.Rodrigues(rvec)
        print "-- rpy --"
        print rpy(rmat)
        tmat = np.zeros((3,4))
        tmat[:3,:3] = rmat
        tmat[:3,3] = tvec.T
        print("-- tmat --")
        print(tmat)
        print "-- projmtx --"
        print np.dot(camMtx, tmat)
        print "-- reproj test --"
        test_reproj(imgPt, objPtMarker.T, tmat, camMtx)
        return tmat, rvec, tvec
    return None, None, None

def getFilenameMat(filename):
    """
    :param filename:
    :return:
     a/b\\c.jpg  ->  cache/tmat_a.b.c.jpg.p
    """
    fn = filename.replace("\\", "/")
    fn = fn.replace("/", ".")
    fn = "cache/tmat_" + fn + ".p"
    return fn

def saveMat(filename, tmat = None):
    fn = getFilenameMat(filename)
    if tmat is None:
        tmat, rvec, tvec = getParams(filename)
        if tmat is None: return None
    f = open(fn, "wb")
    pickle.dump(tmat, f, 2)
    f.close()
    return tmat

def loadMat(filename, noload = True):
    from os.path import isfile
    fn = getFilenameMat(filename)
    if isfile(fn):
        f = open(fn, "rb")
        retval = pickle.load(f)
        f.close()
        return retval
    if noload:
        return None
    else:
        return saveMat(filename)

if __name__ == "__main__":
    print getFilenameMat("a/b\\c.jpg")
    # saveMat("imgs/004.jpg")
    # exit()
    #
    # fn = "imgs/005.jpg"
    # print loadMat(fn)
    # for i in range(5, 10):
    #     fn = "imgs/00%d.jpg" % i
    #     saveMat(fn)
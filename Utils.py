import cv2
import numpy as np
from numpy.linalg import inv

"""
calibrated from: "RemoteSurf/out/2016_11_18__11_51_59/*.jpg"
with flags: cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT
"""
camMtx = np.array([[ 1331.49603,     0.,        479.5    ],
 [    0.,       1331.49603,   359.5    ],
 [    0. ,         0. ,         1.     ]] )
dist_coeffs = np.array( [[ 0.08974, -0.82961,  0.00807,  0.00572,  5.60383]])

size = 14.1
sh = size / 2
objPtMarker = np.array([[-sh, -sh, 0],
                     [sh, -sh, 0],
                     [sh, sh, 0],
                     [-sh, sh, 0]], dtype=np.float32).T

def getObjPtMarkerHomogeneous():
    pts = np.ones((4,4))
    pts[:3,:] = objPtMarker
    return pts

def test_reproj(imgpts, objpts, tmat, cmat):
    numpts =  objpts.shape[0]
    proj = np.dot(cmat, tmat)

    c3d = np.zeros((4, numpts))
    c3d[:3,:] = objpts.T
    c3d[3,:] = np.ones((1, numpts))

    reproj = np.dot(proj, c3d)
    for i in range(numpts):
        w = reproj[2, i]
        for j in range(3):
            reproj[j, i] /= w

    errs = np.abs(imgpts.T - reproj[:2,:])
    max_err = np.max(errs)
    avg_err = np.average(errs)
    if(max_err > 20):
        print "WARNING! ---------------------------------------------------"
    print("max_err: ", max_err)
    print("avg_err: ", avg_err)

def drawMatch(img1, img2, pt1, pt2, good = True, scale = 2):
    realscale = 2
    img1 = cv2.pyrDown(img1)
    img2 = cv2.pyrDown(img2)
    if scale == 4:
        realscale = 4
        img1 = cv2.pyrDown(img1)
        img2 = cv2.pyrDown(img2)

    h, w, c = img1.shape
    out = np.zeros((h, w * 2, 3), np.uint8)
    out[:,:w,:] = img1
    out[:,w:,:] = img2

    color = (255, 255, 0) # cyan
    if good == False:
        color = (0, 0, 255)
    elif good == True:
        color = (0, 255, 0)

    p1 = (int(pt1[0] / realscale), int(pt1[1] / realscale))
    p2 = (int(pt2[0] / realscale + w), int(pt2[1] / realscale))
    text = "pt1(%d, %d), pt2(%d, %d)" % (int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1]))
    cv2.putText(out, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.circle(out, p1, 10, color, 1)
    cv2.circle(out, p2, 10, color, 1)
    cv2.line(out, p1, p2, color, 1)
    cv2.imshow("match", out)

def rpy(mat):
    r = np.arctan2(mat[1,0], mat[0,0])
    if r<-np.pi: r+=np.pi
    if r>np.pi: r-=np.pi

    s1 = np.sin(r)
    c1 = np.cos(r)

    s2 = -mat[2,0]
    c2 = s1 * mat[1,0] + c1 * mat[0,0]
    p = np.arctan2(s2, c2)

    s3 = s1 * mat[0,2] - c1*mat[1,2]
    c3 = -s1*mat[0,1] + c1*mat[1,1]
    y = np.arctan2(s3, c3)

    return r, p, y

def getTransform(c, b, a,  tx,  ty,  tz, is4x4 = False):
    roll, pitch, yaw = c, b, a
    s1 = np.sin(roll)
    c1 = np.cos(roll)
    s2 = np.sin(pitch)
    c2 = np.cos(pitch)
    s3 = np.sin(yaw)
    c3 = np.cos(yaw)

    if not is4x4:
        return np.array([
            [c1*c2, c1*s2*s3-s1*c3, c1*s2*c3+s1*s3, tx],
            [s1*c2, s1*s2*s3+c1*c3, s1*s2*c3-c1*s3, ty],
            [-s2,   c2*s3,          c2*c3,          tz]
        ])
    else:
        return np.array([
            [c1 * c2, c1 * s2 * s3 - s1 * c3, c1 * s2 * c3 + s1 * s3, tx],
            [s1 * c2, s1 * s2 * s3 + c1 * c3, s1 * s2 * c3 - c1 * s3, ty],
            [-s2, c2 * s3, c2 * c3, tz],
            [0, 0, 0, 1]
        ])

def drawMatchesOneByOne(img1, img2, kpt1, kpt2, matches, step = 1):
    cv2.namedWindow("match")
    for i in range(0, len(matches), step):
        match = matches[i]
        drawMatch(img1, img2, kpt1[match.queryIdx].pt, kpt2[match.trainIdx].pt, scale=4)
        if 27 == cv2.waitKey():
            break

def getCrossMat(t):
    tx = t[0]
    ty = t[1]
    tz = t[2]
    return np.array(
        [[0, -tz, ty],
         [tz, 0, -tx],
         [-ty, tx, 0]])

def invTrf(tmat):
    tmat4x4 = np.eye(4)
    tmat4x4[:3,:]=tmat
    return inv(tmat4x4)

def cvt_3x4_to_4x4(mat):
    m4x4 = np.eye(4)
    m4x4[:3,:] = mat
    return m4x4

# img_pt1 = [u1, v1, 1]
# np.dot(im_pt2.T, np.dot(F, im_pt1)) == 0
def calcEssentialFundamentalMat(trf1, trf2, cam1 = camMtx, cam2 = camMtx):
    A1 = np.eye(4)
    A2 = np.eye(4)
    A1[:3, :4] = trf1
    A2[:3, :4] = trf2

    trf = np.dot(A1, inv(A2))

    R = trf[:3, :3].T
    t = trf[:3, 3]

    tx = getCrossMat(t)
    E = np.dot(R, tx)
    F = np.dot(np.dot(inv(cam2.T), E), inv(cam1))

    return E, F

def getDistSqFromEpipolarLine(imPt1, imPt2, F):
    pt1 = np.ones((3, 1))
    pt1[0] = imPt1[0]
    pt1[1] = imPt1[1]
    pt1 = pt1.T
    n = np.dot(pt1, F.T).T
    nx, ny, nz = n[0], n[1], n[2]

    pt2 = imPt2
    dist_sq = (nx * pt2[0] + ny * pt2[1] + nz) ** 2 / (nx * nx + ny * ny)
    return dist_sq

def filterMatchesByEpiline(matches, kpts1, kpts2, F, dist_thr = 20):
    bad_matches = []
    good_matches = []
    while 0 < len(matches):
        match = matches[-1]
        imPt1 = kpts1[match.queryIdx].pt
        imPt2 = kpts2[match.trainIdx].pt
        dsq = getDistSqFromEpipolarLine(imPt1, imPt2, F)
        if dsq > dist_thr ** 2:
            bad_matches.append(matches.pop())
        else:
            good_matches.append(matches.pop())
    return good_matches, bad_matches

def maskKeypoints(masks, kpts):
    num = len(masks)
    print("-- masking --")
    print([len(kpl[1]) for kpl in kpts])
    for i in range(num):
        kp, des = kpts[i]
        j = 0
        while j < len(kp):
            pt = kp[j].pt
            x = int(pt[0])
            y = int(pt[1])
            if masks[i][y, x] > 100:
                j += 1
            else:
                kp.pop(j)
                des.pop(j)
    print([len(kpl[1]) for kpl in kpts])
    return kpts

if __name__ == '__main__':
    print getObjPtMarkerHomogeneous()
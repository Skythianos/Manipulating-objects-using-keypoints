import cv2
import Utils
import numpy as np
from random import random, seed, uniform
from glob import glob
import os
import pickle
from test import calc_rot, calc_trans, calc_avg_rot, reprojectPoints
import DataCache as DC

# outdir = "out0"
# img_points_scale_bad_res = 1600.0 / 640
# pointdict1 = {
#     "../out/%s/0000.jpg": (-250, 0, 200),
#     "../out/%s/0001.jpg": (-250, 0, 250),
#     "../out/%s/0002.jpg": (-250, 0, 300),
#     "../out/%s/0003.jpg": (-300, 0, 300),
#     "../out/%s/0004.jpg": (-200, 0, 300),
#     "../out/%s/0005.jpg": (-250, 50, 300),
#     "../out/%s/0006.jpg": (-250, -50, 300),
#     # "../out/%s/0007.jpg": (),
#     # "../out/%s/0008.jpg": (0, 5, 25),
#     # "../out/%s/0009.jpg": (0, -5, 25),
#     # "../out/%s/0010.jpg": (0, 0, 15),
# }

outdir = "2016_11_7__17_50_40"
pointdict1 = {
    "../out/%s/0000.jpg": (0, 0, 0),
    "../out/%s/0001.jpg": (50, 0, 0),
    "../out/%s/0002.jpg": (-30, 0, 0),
    "../out/%s/0003.jpg": (0, 50, 0),
    "../out/%s/0004.jpg": (0, -50, 0),
    "../out/%s/0005.jpg": (0, 0, -50),
    "../out/%s/0006.jpg": (0, 0, -100),
    "../out/%s/0007.jpg": (0, 0, 50),
    # "../out/%s/0008.jpg": (0, 5, 25),
    # "../out/%s/0009.jpg": (0, -5, 25),
    # "../out/%s/0010.jpg": (0, 0, 15),
}

# outdir = "2016_11_7__18_10_0"
# img_points_scale_bad_res = 1600.0 / 920
# pointdict1 = {
#     "../out/%s/0000.jpg": (-150, 0, 0),
#     "../out/%s/0001.jpg": (-200, 0, 0),
#     "../out/%s/0002.jpg": (-120, 0, 0),
#     "../out/%s/0003.jpg": (-150, -50, 0),
#     "../out/%s/0004.jpg": (-150, -100, 0),
#     "../out/%s/0005.jpg": (-150, 0, -50),
#     "../out/%s/0006.jpg": (-150, 0, -100),
#     "../out/%s/0007.jpg": (-150, -50, -100),
#     # "../out/%s/0008.jpg": (0, 5, 25),
#     # "../out/%s/0009.jpg": (0, -5, 25),
#     # "../out/%s/0010.jpg": (0, 0, 15),
# }

for k in pointdict1:
    pointdict1[k] = map(lambda c: c / 10, pointdict1[k])

cammtx = Utils.camMtx

def normalize(img):
    mmin, mmax = np.min(img), np.max(img)
    return np.uint8((img - mmin) * 255.0 / (mmax - mmin))

def getPts(contours):
    contours = contours[:3]
    cogs = [c.reshape((4, 2)).sum(axis=0) / 4.0 for c in contours]
    up = (cogs[0] + cogs[1]) / 2 - cogs[2]
    right = np.zeros_like(up)
    right[0] = -up[1]
    right[1] = up[0]

    allpts = []
    for c in contours:
        for r in range(c.shape[0]):
            allpts.append(c[r])
    globcog = sum(cogs) / 3
    allpts = [pt - globcog for pt in allpts]
    p00 = max(allpts, key=lambda pt: np.dot(pt, up) + np.dot(pt, -right)) + globcog
    p01 = max(allpts, key=lambda pt: np.dot(pt, up) + np.dot(pt, right)) + globcog
    p10 = max(allpts, key=lambda pt: np.dot(pt, -up) + np.dot(pt, -right)) + globcog
    p11 = max(allpts, key=lambda pt: np.dot(pt, -up) + np.dot(pt, right)) + globcog

    return [p00[0], p01[0], p10[0], p11[0]]

def kabsch(P, Q):
    """
    The optimal rotation matrix U is calculated and then used to rotate matrix
    P unto matrix Q so the minimum root-mean-square deviation (RMSD) can be
    calculated.
    Using the Kabsch algorithm with two sets of paired point P and Q,
    centered around the center-of-mass.
    Each vector set is represented as an NxD matrix, where D is the
    the dimension of the space.
    The algorithm works in three steps:
    - a translation of P and Q
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    http://en.wikipedia.org/wiki/Kabsch_algorithm

    :param P: (N, number of points)x(D, dimension) matrix
    :param Q: (N, number of points)x(D, dimension) matrix
    :return: U -- Rotation matrix
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U

def drawCorners(img, corners):
    img = img.copy()
    for i in range(corners.shape[0]):
        corner = corners[i,0,:]
        cv2.putText(img, str(i), (corner[0], corner[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)
        cv2.circle(img, (corner[0], corner[1]), 10, (0, 0, 255), 2)
    cv2.imshow(" ", img)
    cv2.waitKey()

def img_test():
    """
    AZ IGY KAPOTT MATRIXSZAL A ROBOT KOORDINATA-RENDSZERENEK VEKTORAIT IRJUK FOL A MARKER KOORDINATA RENDSZEREBEN
    :return:
    """
    pointdict = pointdict1

    pattern_size = (9, 6)
    pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
    pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= 2.615

    robot_coords = [pointdict[k] for k in pointdict.keys()]
    imgpts = []

    for k in pointdict.keys():
        k = k % (outdir)
        img = cv2.imread(k)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = normalize(gray)
        rv, corners = cv2.findChessboardCorners(gray, (9, 6))
        # drawCorners(img, corners)
        cv2.cornerSubPix(gray, corners, (9, 6),(-1,-1),criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        # drawCorners(img, corners)

        imgpts_curr = corners.reshape((54,2))
        imgpts.append(imgpts_curr)



    rot, toc = calc_rot(imgpts, pattern_points, robot_coords)

    print Utils.rpy(rot)
    print rot
    # print vrt_np
    # print voc_np
    # print voc_np - vrt_np.dot(rot.T)

    # for i in range(numpts):
    #     for j in range(i + 1, numpts):
    #         print "-", i, j
    #         print np.linalg.norm(vrt_np[i, :] - vrt_np[j, :])
    #         print np.linalg.norm(voc_np[i, :] - voc_np[j, :])

def img_test_from_files(out_dir):
    """
    AZ IGY KAPOTT MATRIXSZAL A ROBOT KOORDINATA-RENDSZERENEK VEKTORAIT IRJUK FOL A MARKER KOORDINATA RENDSZEREBEN
    :return:
    """
    files = glob("%s/*.jpg" % out_dir)


    pattern_size = (9, 6)
    pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
    pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= 2.615

    robot_coords = []
    imgpts = []

    for f in files:
        datafile = os.path.splitext(f)[0] + ".p"
        pfile = file(datafile)
        data = pickle.load(pfile)
        pfile.close()
        robot_coords.append([data[0][i] for i in [500, 501, 502]])

        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = normalize(gray)
        rv, corners = cv2.findChessboardCorners(gray, (9, 6))
        # drawCorners(img, corners)
        cv2.cornerSubPix(gray, corners, (9, 6),(-1,-1),criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        # drawCorners(img, corners)

        imgpts_curr = corners.reshape((54,2))
        imgpts.append(imgpts_curr)



    rot, toc = calc_rot(imgpts, pattern_points, robot_coords)

    print Utils.rpy(rot)
    print rot
    # print vrt_np
    # print voc_np
    # print voc_np - vrt_np.dot(rot.T)

    # for i in range(numpts):
    #     for j in range(i + 1, numpts):
    #         print "-", i, j
    #         print np.linalg.norm(vrt_np[i, :] - vrt_np[j, :])
    #         print np.linalg.norm(voc_np[i, :] - voc_np[j, :])

def img_test_complete_from_files(out_dir, num_rot_calib_imgs, use_calib_data = False):
    file_names_pattern = "%s/*.jpg" % out_dir
    files = glob(file_names_pattern)
    files_rot = files[:num_rot_calib_imgs]

    calib_data_rot = None
    calib_data_trans = None
    if use_calib_data:
        print "using calib data"
        calib_file = "%s/calib_data.p" % out_dir
        calib_data = DC.getData(calib_file)
        if calib_data:
            print "calib_data not None"
            data_len = len(calib_data["tvecs"])
            tvecs = calib_data["tvecs"]
            rvecs = calib_data["rvecs"]
            Utils.camMtx = calib_data["cam_mtx"]
            Utils.dist_coeffs = calib_data["dist_coeffs"]
            calib_data = [(rvecs[i], tvecs[i]) for i in range(data_len)]
            calib_data_rot = calib_data[:num_rot_calib_imgs]
            calib_data_trans = calib_data[num_rot_calib_imgs:]
        else:
            print "ERROR: calib_data is None"

    pattern_size = (9, 6)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= 2.615

    robot_coords_rot = []
    imgpts = []

    a, b, c = -1, -1, -1
    tmats_rt = []
    for f in files_rot:
        datafile = os.path.splitext(f)[0] + ".p"
        pfile = file(datafile)
        data = pickle.load(pfile)
        pfile.close()

        x, y, z, a, b, c = [data[0][i] for i in [500, 501, 502, 503, 504, 505]]
        a, b, c = map(lambda p: p * np.pi / 180     , (a, b, c))  # deg to rad
        x, y, z = map(lambda p: p / 10.0            , (x, y, z))  # mm to cm
        robot_coords_rot.append([x, y, z, a, b, c])
        tmats_rt.append(Utils.getTransform(c, b, a, x, y, z, True))

        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = normalize(gray)
        rv, corners = cv2.findChessboardCorners(gray, (9, 6))
        # drawCorners(img, corners)
        cv2.cornerSubPix(gray, corners, (9, 6), (-1, -1),
                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        # drawCorners(img, corners)

        imgpts_curr = corners.reshape((54, 2))
        imgpts.append(imgpts_curr)

    ror, toc = calc_rot(imgpts, pattern_points, robot_coords_rot, True, calib_data_rot)
    roc = calc_avg_rot([toci[:3,:3] for toci in toc])
    rrt = Utils.getTransform(c, b, a, 0, 0, 0, True)[:3, :3]
    rtc = rrt.T.dot(ror.T.dot(roc))

    print Utils.rpy(ror)
    print ror

    robot_coords_trans = []
    imgpts_trans = []
    tmats_rt_trans = []
    files_trans = files[num_rot_calib_imgs:]
    print [(i, os.path.basename(files_trans[i])) for i in range(len(files_trans))]
    for f in files_trans:
        datafile = os.path.splitext(f)[0] + ".p"
        pfile = file(datafile)
        data = pickle.load(pfile)
        pfile.close()

        x, y, z, a, b, c = [data[0][i] for i in [500, 501, 502, 503, 504, 505]]
        a, b, c = map(lambda p: p * np.pi / 180     , (a, b, c))  # deg to rad
        x, y, z = map(lambda p: p / 10.0            , (x, y, z))  # mm to cm
        robot_coords_trans.append([x, y, z, a, b, c])
        tmats_rt_trans.append(Utils.getTransform(c, b, a, x, y, z, True))

        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = normalize(gray)
        rv, corners = cv2.findChessboardCorners(gray, (9, 6))
        # drawCorners(img, corners)
        cv2.cornerSubPix(gray, corners, (9, 6), (-1, -1),
                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        # drawCorners(img, corners)

        imgpts_curr = corners.reshape((54, 2))
        imgpts_trans.append(imgpts_curr)

    x, toc = calc_trans(imgpts_trans, pattern_points, robot_coords_trans, ror, True, calib_data_trans)
    vtc = x[:3, :]
    vor = x[3:, :]
    tor = np.eye(4)
    tor[:3,:3] = ror
    tor[:3, 3] = vor.reshape((3,))
    ttc = np.eye(4)
    ttc[:3, :3] = rtc
    ttc[:3, 3] = vtc.reshape((3,))

    print "vtc, vor = "
    print x # vtc, vor
    reprojectPoints(
        tor,
        tmats_rt + tmats_rt_trans,
        ttc,
        cammtx,
        pattern_points,
        imgpts + imgpts_trans)
    print "cammtx"
    print Utils.camMtx

    DC.saveData("%s/arrangement_calib.p" % out_dir, {"ttc": ttc, "tor": tor, "cam_mtx": Utils.camMtx, "dist_coeffs": Utils.dist_coeffs})

def filter_contours(contours):
    # print  len(contours)
    # h, w = dil.shape
    # area = h * w
    # contours = [c for c in contours if cv2.contourArea(c) > area / 8]
    # print  len(contours)
    contours = [cv2.approxPolyDP(c, 20, True) for c in contours]
    # print  len(contours)
    contours = [c for c in contours if c.shape[0] == 4]
    # print len(contours)
    contours = [c for c in sorted(contours, key=lambda c: cv2.contourArea(c))]
    contours = [c for c in contours if cv2.contourArea(c) > 5000]
    return contours


# todo:
# -check out other calib methods
# -debug arrangement_calib/test/test(), see:
#     # img_pts = [img_pts[i] for i in range(len(img_pts)) if i != 776]
#     # robot_coords = [robot_coords[i] for i in range(len(robot_coords)) if i != 776]

def test():
    seed(0)
    num_imgs = 10
    num_obj_pts = 20
    obj_pts = np.random.random((3, num_obj_pts))
    obj_pts_homog = np.ones((4, num_obj_pts))
    obj_pts_homog[:3, :] = obj_pts

    tmats_rt = [None] * num_imgs
    r, p, y = .1, .2, .3
    # r, p, y = 0, 0, 0
    img_pts = [None] * num_imgs
    robot_coords = [None] * num_imgs

    tmat_or = Utils.getTransform(1.1, .2, .3, 0, 0, 0, True)
    tmat_tc = Utils.getTransform(0, 0, 0, 0, 0, 0, True)

    for i in range(num_imgs):
        robot_coords[i] = map(int, (uniform(-1, 1) * 10, uniform(-1, 1) * 10, uniform(-1, 1) * 10))
        tmats_rt[i] =  Utils.getTransform(r, p, y, robot_coords[i][0], robot_coords[i][1], robot_coords[i][2], True)
        # print tmats_rt[i]
        tmat_oc = tmat_or.dot(tmats_rt[i].dot(tmat_tc))
        tmat_co = np.linalg.inv(tmat_oc)
        # print tmat_co
        cam_pts = tmat_co.dot(obj_pts_homog)
        proj_pts = cammtx.dot(cam_pts[:3, :])
        for j in range(num_obj_pts):
            proj_pts[:, j] /= proj_pts[2, j]
        img_pts[i] = proj_pts[:2, :]

    rot, toc = calc_rot(img_pts, obj_pts, robot_coords)
    print "---"
    print rot
    print tmat_or

if __name__ == '__main__':
    # np.set_printoptions(formatter={'float': lambda x: "\t{0:0.3f}".format(x)})
    np.set_printoptions(precision=3, suppress=True)


    # out_dir = "../out/2016_11_15__15_2_40"
    # out_dir = "../out/2016_11_15__15_35_17"
    # out_dir = "../out/2016_11_15__15_39_53"
    # img_test_from_files(out_dir)


    out_dir = "../out/2017_5_19__17_40_11"
    img_test_complete_from_files(out_dir, 34, True)



    # img_test()
    # test()

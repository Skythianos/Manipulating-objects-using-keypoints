import cv2
import numpy as np
import cPickle as pickle
from os.path import isfile
import Utils as util
import MarkerDetect as MD

MATCHER_FLANN_RATIO_07 = "flann_ratio"
MATCHER_BF_RATIO_07 = "bf_ratio"
MATCHER_BF_CROSS = "bf_cross"
MATCHER_BF_CROSS_EPILINES = "bf_cross_epilines"
MATCHER_BF_CROSS_EPILINES_AFTER = "bf_cross_epilines_after"
MATCHER_BF_MULTIPLE = "bf_multiple"
MATCHER_BF_EPILINES_HOMOGRAPHY = "bf_epilines_homography"

# dump file name: "cache/filename.detectorType.matcherType.version.p"
class MatchLoader:
    def getFileName(self, fn1, fn2, dType, mType, version):
        if fn1.startswith("out"):
            fn1 = fn1.replace("\\", "/").replace("/", "_")
            fn2 = fn2.replace("\\", "/").replace("/", "_")
            return "cache/match_%s.%s.%s.%s.%s.p" % (fn1, fn2, dType, mType, version)
        fn1 = fn1[fn1.rindex("/") + 1:]
        fn2 = fn2[fn2.rindex("/") + 1:]
        return "cache/match_%s.%s.%s.%s.%s.p" % (fn1, fn2, dType, mType, version)

    def loadMatches(self, filename1, filename2, detectorType, matcherType, version):
        detectorType = detectorType.lower()
        matcherType = matcherType.lower()

        fname = self.getFileName(filename1, filename2, detectorType, matcherType, version)
        if isfile(fname):
            f = open(fname, "rb")
            matches = pickle.load(f)
            f.close()
            return fname, self.deserializeMatches(matches)

        return fname, None

    def matchBFCross(self, filename1, filename2, des1, des2, detectorType, version = "0", noload = False, nosave = False):
        fname, matches = self.loadMatches(filename1, filename2, detectorType, MATCHER_BF_RATIO_07, version)
        if matches is not None and not noload:
            return matches

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors.
        good = bf.match(np.asarray(des1, np.float32), np.asarray(des2, np.float32))

        if not nosave:
            f = open(fname, "wb")
            pickle.dump(self.serializeMatches(good), f, 2)
            f.close()

        return good

    def matchBFRatio(self, filename1, filename2, des1, des2, detectorType, ratio = 0.7, version = "0", noload = False, nosave = False):
        fname, matches = self.loadMatches(filename1, filename2, detectorType, MATCHER_BF_RATIO_07, version)
        if matches is not None and not noload:
            return matches

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Match descriptors.
        matches = bf.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)

        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)

        good.sort(key=lambda x: x.distance, reverse=True)

        if not nosave:
            f = open(fname, "wb")
            pickle.dump(self.serializeMatches(good), f, 2)
            f.close()

        return good

    def matchBFCrossEpilinesMultiple(self, filename1, filename2, des1, des2, kpts1, kpts2,
                             tmat1, tmat2, detectorType, match_per_point = 4, step = 1, version = "0", noload = False, nosave = False):
        fname, matches = self.loadMatches(filename1, filename2, detectorType, MATCHER_BF_MULTIPLE, version)
        if matches is not None and not noload:
            return matches

        # assert False #implementetion not finished

        num1, num2 = len(kpts1), len(kpts2)
        E, F = util.calcEssentialFundamentalMat(tmat1, tmat2)

        dist_thr = 10 ** 2 #distance threshold from epiline

        match1 = self._match_epilines_multiple_inner(F, des1, des2, dist_thr, kpts1, kpts2, step, match_per_point)
        match2 = self._match_epilines_multiple_inner(F.T, des2, des1, dist_thr, kpts2, kpts1, step, match_per_point, reverse=True)

        #cross-check
        match1 = [(m.queryIdx, m.trainIdx) for m in match1]
        match2 = [(m.queryIdx, m.trainIdx) for m in match2]
        s1 = set(match1)
        s2 = set(match2)
        isec = s1.intersection(s2)

        all_matches = [cv2.DMatch(
                    _queryIdx=gmatch[0],
                    _trainIdx=gmatch[1],
                    _imgIdx=0,
                    _distance=-1) for gmatch in isec]

        if not nosave:
            f = open(fname, "wb")
            pickle.dump(self.serializeMatches(all_matches), f, 2)
            f.close()

        return all_matches

    def _match_epilines_multiple_inner(self, F, des1, des2, dist_thr, kpts1, kpts2, step, match_per_point, reverse=False):
        num1, num2 = len(kpts1), len(kpts2)
        match1 = []

        kpt2_mat = np.ones((3, num2))
        for i in range(num2):
            kpt2_mat[0, i] = kpts2[i].pt[0]
            kpt2_mat[1, i] = kpts2[i].pt[1]

        for i in range(0, num1, step):
            if i % 100 == 0: print i, num1
            pt1 = kpts1[i].pt

            pt1_h = np.ones((3, 1))
            pt1_h[0] = pt1[0]
            pt1_h[1] = pt1[1]
            pt1_h = pt1_h.T
            n = np.dot(pt1_h, F.T).T
            nx, ny, nz = n[0], n[1], n[2]

            thr = np.sqrt((nx * nx + ny * ny) * dist_thr)
            distances = np.abs(n.T.dot(kpt2_mat))
            distances = distances.reshape(num2)
            idx_list = np.where(distances < thr)[0]

            des_list1 = [des1[i]]
            des_list2 = [des2[j] for j in idx_list]

            good = []
            if len(des_list2) > 0:
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                matches = bf.knnMatch(np.asarray(des_list1, np.float32), np.asarray(des_list2, np.float32), k=match_per_point)

                good = []
                for m in matches:
                    good.extend(list(m))

            if not reverse:
                match1.extend([cv2.DMatch(
                    _queryIdx=i,
                    _trainIdx=idx_list[gmatch.trainIdx],
                    _imgIdx=0,
                    _distance=gmatch.distance) for gmatch in good])
            else:
                match1.extend([cv2.DMatch(
                    _queryIdx=idx_list[gmatch.trainIdx],
                    _trainIdx=i,
                    _imgIdx=0,
                    _distance=gmatch.distance) for gmatch in good])

        return match1


    def matchBFCrossEpilines(self, filename1, filename2, des1, des2, kpts1, kpts2,
                             tmat1, tmat2, detectorType, step = 1, version = "0", noload = False, nosave = False):
        fname, matches = self.loadMatches(filename1, filename2, detectorType, MATCHER_BF_CROSS_EPILINES, version)
        if matches is not None and not noload:
            return matches

        E, F = util.calcEssentialFundamentalMat(tmat1, tmat2)

        dist_thr = 10 ** 2 #distance threshold from epiline

        match1 = self._match_epilines_inner(F, des1, des2, dist_thr, kpts1, kpts2, step)
        match2 = self._match_epilines_inner(F.T, des2, des1, dist_thr, kpts2, kpts1, step, reverse=True)

        #cross-check
        match1 = [(m.queryIdx, m.trainIdx) for m in match1]
        match2 = [(m.queryIdx, m.trainIdx) for m in match2]
        s1 = set(match1)
        s2 = set(match2)
        isec = s1.intersection(s2)

        all_matches = [cv2.DMatch(
                    _queryIdx=gmatch[0],
                    _trainIdx=gmatch[1],
                    _imgIdx=0,
                    _distance=-1) for gmatch in isec]

        if not nosave:
            f = open(fname, "wb")
            pickle.dump(self.serializeMatches(all_matches), f, 2)
            f.close()

        return all_matches

    def matchBFEpilinesHomogr(self, filename1, filename2, des1, des2, kpts1, kpts2,
                             tmat1, tmat2, detectorType, step = 1, version = "0", noload = False, nosave = False):
        fname, matches = self.loadMatches(filename1, filename2, detectorType, MATCHER_BF_EPILINES_HOMOGRAPHY, version)
        if matches is not None and not noload:
            return matches

        E, F = util.calcEssentialFundamentalMat(tmat1, tmat2)

        dist_thr = 10 ** 2 #distance threshold from epiline

        match1 = self._match_epilines_inner(F, des1, des2, dist_thr, kpts1, kpts2, step)
        match2 = self._match_epilines_inner(F.T, des2, des1, dist_thr, kpts2, kpts1, step, reverse=True)

        #union
        match1 = [(m.queryIdx, m.trainIdx) for m in match1]
        match2 = [(m.queryIdx, m.trainIdx) for m in match2]
        s1 = set(match1)
        s2 = set(match2)
        all = s1.union(s2)

        srcPts = np.float32([kpts1[m[0]].pt for m in all]).reshape(-1,1,2)
        dstPts = np.float32([kpts2[m[1]].pt for m in all]).reshape(-1,1,2)

        retval, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)

        all = list(all)
        good = []
        for i in range(len(all)):
            if mask[i][0] == 1:
                good.append(all[i])

        all_matches = [cv2.DMatch(
                    _queryIdx=gmatch[0],
                    _trainIdx=gmatch[1],
                    _imgIdx=0,
                    _distance=-1) for gmatch in good]

        if not nosave:
            f = open(fname, "wb")
            pickle.dump(self.serializeMatches(all_matches), f, 2)
            f.close()

        return all_matches

    def _match_epilines_inner(self, F, des1, des2, dist_thr, kpts1, kpts2, step, reverse=False):
        num1, num2 = len(kpts1), len(kpts2)
        match1 = []

        kpt2_mat = np.ones((3, num2))
        for i in range(num2):
            kpt2_mat[0, i] = kpts2[i].pt[0]
            kpt2_mat[1, i] = kpts2[i].pt[1]

        for i in range(0, num1, step):
            if i % 100 == 0: print i, num1
            pt1 = kpts1[i].pt

            pt1_h = np.ones((3, 1))
            pt1_h[0] = pt1[0]
            pt1_h[1] = pt1[1]
            pt1_h = pt1_h.T
            n = np.dot(pt1_h, F.T).T
            nx, ny, nz = n[0], n[1], n[2]

            thr = np.sqrt((nx * nx + ny * ny) * dist_thr)
            distances = np.abs(n.T.dot(kpt2_mat))
            distances = distances.reshape(num2)
            idx_list = np.where(distances < thr)[0]

            des_list1 = [des1[i]]
            des_list2 = [des2[j] for j in idx_list]

            good = []
            if len(des_list2) > 0:
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                good = bf.match(
                    np.asarray(des_list1, np.float32), np.asarray(des_list2, np.float32))

            if not reverse:
                match1.extend([cv2.DMatch(
                    _queryIdx=i,
                    _trainIdx=idx_list[gmatch.trainIdx],
                    _imgIdx=0,
                    _distance=gmatch.distance) for gmatch in good])
            else:
                match1.extend([cv2.DMatch(
                    _queryIdx=idx_list[gmatch.trainIdx],
                    _trainIdx=i,
                    _imgIdx=0,
                    _distance=gmatch.distance) for gmatch in good])

        return match1

    def matchBFCrossEpilinesAfter(self, filename1, filename2, des1, des2, kpts1, kpts2,
                             tmat1, tmat2, detectorType, dist_thr = 20, version="0", noload=False, nosave=False):
        fname, matches = self.loadMatches(filename1, filename2, detectorType, MATCHER_BF_CROSS_EPILINES_AFTER, version)
        if matches is not None and not noload:
            return matches

        E, F = util.calcEssentialFundamentalMat(tmat1, tmat2)
        matches = self.matchBFCross(filename1, filename2, des1, des2, detectorType, version)
        good, bad = util.filterMatchesByEpiline(matches, kpts1, kpts2, F, dist_thr)

        if not nosave:
            f = open(fname, "wb")
            pickle.dump(self.serializeMatches(good), f, 2)
            f.close()

        return good

    def matchFLANNRatio(self, filename1, filename2, des1, des2, detectorType, ratio = 0.7, version = "0", noload = False, nosave = False):
        fname, matches = self.loadMatches(filename1, filename2, detectorType, MATCHER_BF_RATIO_07, version)
        if matches is not None and not noload:
            return matches

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)

        good.sort(key=lambda x: x.distance, reverse=True)

        if not nosave:
            f = open(fname, "wb")
            pickle.dump(self.serializeMatches(good), f, 2)
            f.close()

        return good

    def serializeMatches(self, matches):
        return [(m.distance, m.imgIdx, m.queryIdx, m.trainIdx) for m in matches]

    def deserializeMatches(self, serd):
        return [
            cv2.DMatch(_queryIdx = m[2], _trainIdx = m[3], _imgIdx = m[1], _distance = m[0])
            for m in serd]

def print_rand(arr, idxs):
    sel = [arr[idx] for idx in idxs]
    print(sel)

def drawMatches(img1, img2, pt1, pt2, scale1 = 1, scale2 = 1, num = 10, skip = 0):
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape
    img1 = cv2.resize(img1, tuple(map(int, (w1 / scale1, h1 / scale1))))
    img2 = cv2.resize(img2, tuple(map(int, (w2 / scale2, h2 / scale2))))
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape

    idx = 0
    numMatches = len(pt1)

    while True:
        if idx >= numMatches:
            break

        out = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        out[:h1, :w1, :] = img1
        out[:h2, w1:, :] = img2

        for i in range(num):
            if idx >= numMatches:
                break
            p1 = (int(pt1[idx][0] / scale1), int(pt1[idx][1] / scale1))
            p2 = (int(pt2[idx][0] / scale2 + w1), int(pt2[idx][1] / scale2))
            # text = "pt1(%d, %d), pt2(%d, %d)" % (int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1]))
            # cv2.putText(out, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            color = (255 - (255 * i) / num, (255 * i) / num, (255 * i) / num)
            cv2.circle(out, p1, 10, color, 1)
            cv2.circle(out, p2, 10, color, 1)
            cv2.line(out, p1, p2, color, 1)
            idx += 1 + skip

        cv2.imshow("img1", out)
        cv2.waitKey()

def navigate():
    files = glob("out/2017_3_8__14_51_22/*.jpg")
    imgs = [cv2.imread(f) for f in files]
    keys = {"w": 119,
            "a": 97,
            "s": 115,
            "d": 100,
            "esc": 27}
    idx = 0
    while True:
        cv2.imshow("img", imgs[idx])
        print idx
        key = cv2.waitKey()
        newidx = idx
        if key == keys["a"]:
            newidx = idx - 7
        elif key == keys["s"]:
            newidx = idx - 1
        elif key == keys["d"]:
            newidx = idx + 7
        elif key == keys["w"]:
            newidx = idx + 1
        elif key == keys["esc"]:
            break
        if 0 <= newidx < len(imgs):
            idx = newidx

def find_corners():
    objp_total = []
    imgpt_total = []
    files_dir = "out/2017_4_5__15_31_34/"
    files_dir = "out/2017_4_5__15_57_20/"
    # files_dir = "out/2017_3_8__14_51_22/"
    files = glob(join(files_dir, "*.jpg"))

    grid_size = (3, 6)
    real_size = 2.615
    objp_right = np.float32(np.mgrid[0:0 + grid_size[0], 0: grid_size[1]].T.reshape(-1, 2))
    objp_right_h = np.ones((grid_size[0] * grid_size[1], 3), np.float32)
    objp_right_h[:, :2] = objp_right
    objp_left = np.float32(np.mgrid[6:6 + grid_size[0], 0:grid_size[1]].T.reshape(-1, 2))
    objp_left_h = np.ones((grid_size[0] * grid_size[1], 3), np.float32)
    objp_left_h[:, :2] = objp_left
    objp_all = np.zeros((36, 3))
    objp_all[18:, :2] = objp_left
    objp_all[:18, :2] = objp_right
    objp_all *= real_size

    for f in files:
        print f
        img_orig = cv2.imread(f, 0)

        img_color = cv2.imread(f)
        scale = 2
        h, w = img_orig.shape
        img = cv2.resize(img_orig, (w / scale, h / scale))
        img_color = cv2.resize(img_color, (w / scale, h / scale))
        h, w = img.shape
        offset = 0
        cut = w / 2 + offset
        img_left = img[:, :cut]
        img_right = img[:, cut:]
        cv2.imshow("left", img_left)
        cv2.imshow("right", img_right)

        rret, rcorners = cv2.findChessboardCorners(img_right, grid_size,
                                                 flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        lret, lcorners = cv2.findChessboardCorners(img_left, grid_size,
                                                 flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not lret and not rret:
            print "ERR"
            continue
        print f

        left = lret
        print "left: ", left
        if left:
            ret, corners = lret, lcorners
        else:
            ret, corners = rret, rcorners

        corners = np.fliplr(np.flipud(corners))
        if not left:
            corners = corners.reshape(-1, 2)
            corners[:, 0] = corners[:, 0] + np.ones((18,)) * cut
        corners = corners.reshape(-1, 1, 2)

        if ret:
            corners_orig = corners * scale
            cv2.cornerSubPix(img_orig, corners_orig, (11, 11), (-1, -1),
                             criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            corners = corners_orig.reshape(-1, 2) / scale
            cv2.drawChessboardCorners(img_color, grid_size, corners, ret)
            idx = 0
            offset = 0 if not left else len(corners)
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    cv2.putText(img_color, str(idx + offset), tuple(corners[idx, :]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                    idx += 1

        if ret:
            if not left:
                H, mask = cv2.findHomography(objp_right, corners.reshape(-1, 2))
                corners2 = np.float32(H.dot(objp_left_h.T)).T
                corners2 = cv2.convertPointsFromHomogeneous(corners2).reshape((-1, 2))
            else:
                H, mask = cv2.findHomography(objp_left, corners.reshape(-1, 2))
                corners2 = np.float32(H.dot(objp_right_h.T)).T
                corners2 = cv2.convertPointsFromHomogeneous(corners2).reshape((-1, 2))

        if ret:
            corners2_orig = corners2 * scale
            cv2.cornerSubPix(img_orig, corners2_orig, (11, 11), (-1, -1),
                             criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            corners2 = corners2_orig / scale
            cv2.drawChessboardCorners(img_color, grid_size, corners2, ret)
            idx = 0
            offset = 0 if left else len(corners)
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    cv2.putText(img_color, str(idx + offset), tuple(corners2[idx, :]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                    idx += 1

        corners_all = np.zeros((36, 2))
        if not left:
            corners_all[:18, :] = corners_orig.reshape(-1, 2)
            corners_all[18:, :] = corners2_orig.reshape(-1, 2)
        else:
            corners_all[:18, :] = corners2_orig.reshape(-1, 2)
            corners_all[18:, :] = corners_orig.reshape(-1, 2)

        objp_total.append(objp_all)
        imgpt_total.append(corners_all)

        retval, rvec, tvec = cv2.solvePnP(objp_all, corners_all, Utils.camMtx, Utils.dist_coeffs, flags=cv2.ITERATIVE)
        # print objp_all
        # print corners_all
        datafilename = f.replace("\\", "/").replace("/", "_")
        f_handle = open("cache/points_%s.p" % datafilename, "wb")
        pickle.dump({"objp": objp_all, "imgp": corners_all, "rvec": rvec, "tvec": tvec}, f_handle)
        f_handle.close()
        # print rvec, tvec

        #0 15 20 35
        # cv2.putText(img_color, str("%.2f, %.2f, %.2f" % tuple(objp_all[0,:])), tuple(np.int32(corners_all[0, :]/2)), cv2.FONT_HERSHEY_SIMPLEX,                    0.5, (0, 0, 255))
        # cv2.putText(img_color, str("%.2f, %.2f, %.2f" % tuple(objp_all[15,:])), tuple(np.int32(corners_all[15, :]/2)), cv2.FONT_HERSHEY_SIMPLEX,                    0.5, (0, 0, 255))
        # cv2.putText(img_color, str("%.2f, %.2f, %.2f" % tuple(objp_all[20,:])), tuple(np.int32(corners_all[20, :]/2)), cv2.FONT_HERSHEY_SIMPLEX,                    0.5, (0, 0, 255))
        # cv2.putText(img_color, str("%.2f, %.2f, %.2f" % tuple(objp_all[35,:])), tuple(np.int32(corners_all[35, :]/2)), cv2.FONT_HERSHEY_SIMPLEX,                    0.5, (0, 0, 255))
        cv2.imshow("asd", img_color)
        # cv2.waitKey()
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp_total, imgpt_total, (960, 720),
    #                                                    flags=cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT)
    # print mtx

def getTmat(fname):
    import pickle
    from os.path import isfile

    datafilename = fname.replace("\\", "/").replace("/", "_")
    datafilename = "cache/points_%s.p" % datafilename
    if not isfile(datafilename):
        return None
    f_handle = open(datafilename, "rb")
    dct = pickle.load(f_handle) #{"objp": objp_all, "imgp": corners_all, "rvec": rvec, "tvec": tvec}, f_handle)
    f_handle.close()

    rvec = dct["rvec"]
    tvec = dct["tvec"]

    rmat, jac = cv2.Rodrigues(rvec)
    tmat = np.zeros((3, 4))
    tmat[:3, :3] = rmat
    tmat[:3, 3] = tvec.T
    return tmat

def match_pairs():
    #BF epilines homogr (doksi: 7.4.3)

    files_dir = "out/2017_3_8__14_51_22/"
    files = glob(join(files_dir, "*.jpg"))

    pairs = []
    for i in range(7):
        for j in range(7):
            img_idx = i * 7 + j
            pairs.append((img_idx, img_idx + 7))
            pairs.append((img_idx, img_idx - 7))
            pairs.append((img_idx, img_idx + 1))
            pairs.append((img_idx, img_idx - 1))
    pairs = [p for p in pairs if 0 <= p[1] < len(files)]
    # pprint(pairs)
    for pair in pairs:
        # fname1 = "out/2017_3_8__14_51_22/0000.jpg"
        # fname2 = "out/2017_3_8__14_51_22/0007.jpg"
        # fname2 = "imgs/004.jpg"

        fname1 = files[pair[0]]
        fname2 = files[pair[1]]
        mask1 = fname1.replace(".jpg", "_mask.png")
        mask2 = fname2.replace(".jpg", "_mask.png")
        mask1 = cv2.imread(mask1, 0)
        mask2 = cv2.imread(mask2, 0)

        img1 = cv2.imread(fname1)
        img2 = cv2.imread(fname2)

        fl = FL.FeatureLoader()
        kp1, des1 = fl.loadFeatures(fname1)
        kp2, des2 = fl.loadFeatures(fname2)
        if len(des1) == 0 or len(des2) == 0:
            continue

        tmat1 = getTmat(fname1)
        tmat2 = getTmat(fname2)

        res = Utils.maskKeypoints([mask1, mask2], [(kp1, des1), (kp2, des2)])
        kp1, des1 = res[0]
        kp2, des2 = res[1]

        if False:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            match1 = bf.match(np.asarray(des1, np.float32), np.asarray(des2, np.float32))
            match2 = bf.match(np.asarray(des2, np.float32), np.asarray(des1, np.float32))

            # union
            match1 = [(m.queryIdx, m.trainIdx) for m in match1]
            match2 = [(m.trainIdx, m.queryIdx) for m in match2]
            s1 = set(match1)
            s2 = set(match2)
            all = s1.union(s2)

            srcPts = np.float32([kp1[m[0]].pt for m in all]).reshape(-1, 1, 2)
            dstPts = np.float32([kp2[m[1]].pt for m in all]).reshape(-1, 1, 2)

            retval, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)

            all = list(all)
            good = []
            for i in range(len(all)):
                if mask[i][0] == 1:
                    good.append(all[i])

            all_matches = [cv2.DMatch(
                _queryIdx=gmatch[0],
                _trainIdx=gmatch[1],
                _imgIdx=0,
                _distance=-1) for gmatch in good]
        else:
            ml = MatchLoader()
            all_matches = ml.matchBFEpilinesHomogr(fname1, fname2, des1, des2, kp1, kp2, tmat1, tmat2, "surf")

        print fname1, fname2
        print "all matches:"
        print len(all_matches)
        pt1 = [kp1[m.queryIdx].pt for m in all_matches]
        pt2 = [kp2[m.trainIdx].pt for m in all_matches]
        drawMatches(img1, img2, pt1, pt2, 2, 2, 30, 10)


        # fl = FL.FeatureLoader()
        #
        # fn1 = "imgs/005.jpg"
        # fn2 = "imgs/006.jpg"
        # img1 = cv2.imread(fn1)
        # img2 = cv2.imread(fn2)
        # kp1, des1 = fl.loadFeatures(fn1, "SURF")
        # kp2, des2 = fl.loadFeatures(fn2, "SURF")
        # print(len(des1), len(des2))
        #
        # tmat1 = MD.loadMat(fn1)
        # tmat2 = MD.loadMat(fn2)
        #
        # ml = MatchLoader()
        # m, b = ml.matchBFEpilinesHomogr(fn1, fn2, des1, des2, kp1, kp2, tmat1, tmat2, "surf", nosave=True)
        # print len(m), len(b)
        # util.drawMatchesOneByOne(img1, img2, kp1, kp2, m, 1)
        # print "bad"
        # util.drawMatchesOneByOne(img1, img2, kp1, kp2, b, 50)

def create_pos_cache():
    files_dir = "out/2017_3_8__14_51_22/"
    files_dir = "out/2017_4_5__15_31_34/"
    files_dir = "out/2017_4_5__15_57_20/"
    files = glob(join(files_dir, "*.jpg"))
    masks = []
    for f in files:
        m = f.replace(".jpg", "_mask.png")
        masks.append(m)

    for f in files:
        if getTmat(f) is not None:
            MarkerDetect.saveMat(f, getTmat(f))


def test():
    from SFMSolver import draw_real_coords
    files_dir = "out/2017_3_8__14_51_22/"
    files = glob(join(files_dir, "*.jpg"))
    masks = []
    for f in files:
        m = f.replace(".jpg", "_mask.png")
        masks.append(m)
    sfm = SFM.SFMSolver(files, masks)

    # pointData is list of tuple: (des, p3d, img_idx, kpt_idx)
    imgs, kpts, points, pointData = sfm.calc_data_from_files_triang_simple()

    imidx = 1
    skip = 100
    pdata0 = [p for p in pointData if p[2] == imidx]
    print len(pdata0)
    pts3d = [p3d for _, p3d, _, kpidx in pdata0]
    impts = [kpts[imidx][0][kpidx].pt for _, _, _, kpidx in pdata0]
    pts3d = [pts3d[i] for i in range(0, len(pts3d), skip)]
    impts = [impts[i] for i in range(0, len(impts), skip)]
    draw_real_coords(imgs[imidx], impts, pts3d, True)

if __name__ == "__main__":
    import FeatureLoader as FL
    import random
    from os.path import join
    from pprint import pprint
    import Utils
    from glob import glob
    import DataCache as DC
    import SFMSolver as SFM
    import MarkerDetect

    # navigate()

    find_corners()

    # match_pairs()

    create_pos_cache()


    # test()
    # find_corners()
    # create_pos_cache()
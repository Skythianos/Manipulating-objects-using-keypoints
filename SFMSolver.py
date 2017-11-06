import cv2
import numpy as np
import FeatureLoader
import MatchLoader
import Utils
import MarkerDetect
from pprint import pprint
import DataCache as DC

class CliqueExtractor:
    def getCliques(self, graph, max_num):
        triplets = []
        for k in graph:
            neigh = list(graph[k])
            num_n = len(neigh)
            for i in range(num_n):
                for j in range(i + 1, num_n):
                    n1 = neigh[i]
                    n2 = neigh[j]
                    if k in graph[n1] and k in graph[n2]:
                        triplets.append((k, n1, n2))
        num = 3
        cliques = triplets
        all_levels = [triplets]
        while num < max_num:
            cliques = self._add_level(graph, cliques)
            all_levels.append(list(cliques))
            num += 1
        return all_levels

    def _add_level(self, graph, cliques):
        newcliques = set()
        for c in cliques:
            elem = c[0]
            for n in graph[elem]:
                if self._connected_to_all(graph, n, c):
                    newclique = list(c)
                    newclique.append(n)
                    newcliques.add(tuple(sorted(newclique)))
        return newcliques

    def _connected_to_all(self, graph, node, clique):
        for n in clique:
            if node not in graph[n]:
                return False
        return True

class SFMSolver:
    def __init__(self, filenames, masks):
        self.filenames = filenames
        self.masks = [cv2.imread(m, 0) for m in masks]
        self.detector = "surf"
        assert masks is None or len(filenames) == len(masks)

    def getMatches(self):
        print("-- load features --")
        files = self.filenames
        num = len(files)

        fl = FeatureLoader.FeatureLoader()
        ml = MatchLoader.MatchLoader()
        kpts = [fl.loadFeatures(f, self.detector) for f in files]
        tmats = [MarkerDetect.loadMat(f) for f in files]

# masking
        kpts = self.maskKeypoints(kpts)

# match
        print("-- matching --")
        print("num imgs: %d" % num)
        matches = [[None] * num for i in range(num)]
        for i in range(num):
            # print(i)
            for j in range(num):
                if i == j: continue

                # matches[i][j] = ml.matchBFCrossEpilines(
                #     self.filenames[i],
                #     self.filenames[j],
                #     kpts[i][1],
                #     kpts[j][1],
                #     kpts[i][0],
                #     kpts[j][0],
                #     tmats[i],
                #     tmats[j],
                #     "surf"
                # )

                # matches[i][j] = ml.matchBFEpilinesHomogr(
                #     self.filenames[i],
                #     self.filenames[j],
                #     kpts[i][1],
                #     kpts[j][1],
                #     kpts[i][0],
                #     kpts[j][0],
                #     tmats[i],
                #     tmats[j],
                #     "surf"
                # )

                fn, curr_matches = ml.loadMatches(
                    self.filenames[i],
                    self.filenames[j],
                    "surf",
                    MatchLoader.MATCHER_BF_EPILINES_HOMOGRAPHY,
                    0
                )
                matches[i][j] = curr_matches
                if curr_matches is None:
                    matches[i][j] = []
                # else:
                #     print self.filenames[i], self.filenames[j], len(curr_matches)

        sm = 0
        for i in range(num):
            for j in range(num):
                if matches[i][j] is not None:
                    sm += len(matches[i][j])
        print "sum: ", sm

        return matches, kpts

    def maskKeypoints(self, kpts):
        num = len(self.filenames)
        masks = self.masks
        if masks is not None:
            print("-- masking --")
            print(len(kpts[0][1]))
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
            print(len(kpts[0][1]))
        return kpts

    def getGraph(self, matches, kpts):
# graph
        print("-- graph --")
        num = len(self.filenames)
        graph = {}

        for i in range(num):
            for j in range(num):
                if i == j: continue
                for m in matches[i][j]:
                    id1 = (i, m.queryIdx)
                    id2 = (j, m.trainIdx)
                    if id1 not in graph:
                        graph[id1] = set()
                    graph[id1].add(id2)

#both ways
        print("graph size: %d" % len(graph))
        for k in graph.keys():
            for v in list(graph[k]):
                if v not in graph:
                    graph[v] = set()
                graph[v].add(k)
        print("graph size: %d" % len(graph))

#connectivity print
        print("-- connectivity --")
        total = 0
        i = 0
        while True:
            subg = [(k, graph[k]) for k in graph if len(graph[k]) == i]
            print(i, len(subg))
            total += len(subg)
            i += 1
            if total == len(graph):
                break
        return graph

    def extractCliques(self, graph, maxlevel = 5):
        print("levels")
        cliqueExtr = CliqueExtractor()
        all_levels = cliqueExtr.getCliques(graph, maxlevel)
        for i in range(len(all_levels)):
            level = all_levels[i]
            print(i + 3, len(level))

        return all_levels

    def extendCliques(self, graph, cliques, max_missing = 1):
        for i in range(len(cliques)):
            if i % 1000 == 0: print i, len(cliques)
            clique = cliques[i]
            clique = set(clique)
            num = len(clique)
            for node in clique:     #for each node
                missing = []
                num_connected = 0
                for k in list(graph[node]):   #for each neighbor k of node
                    if k in clique:
                        continue
                    missing = clique - graph[k]
                    if len(missing) <= max_missing:
                        #add missing edges to k's neigh list
                        graph[k].update(missing)
                        for n in clique:
                            graph[n].add(k)

    def print_graph(self, graph, node, depth):
        nodes = self._get_subgraph(graph, node, depth)
        for n in nodes:
            print n, graph[n]

    def _get_subgraph(self, graph, node, depth):
        if depth == 0:
            return [node]

        nodes = set()
        nodes.add(node)
        for neigh in graph[node]:
            nnodes = self._get_subgraph(graph, neigh, depth - 1)
            for n in nnodes:
                nodes.add(n)
        return list(nodes)

    def getCliquePosRANSAC(self, clique, kpts, tmats, min_inliers=3, err_thresh=200):
        num = len(clique)
        pos = [[None] * num for n in clique]
        projMats = [np.dot(Utils.camMtx, tmat) for tmat in tmats]
        imgPts = [np.array(kpts[imgidx][0][kptidx].pt, dtype="float32")
                  for imgidx, kptidx in clique]
        res = [[None] * num for n in clique]

        best = None
        besterr = 0
        for i in range(num):
            for j in range(i + 1, num):
                p4d = self._triangulate(
                    projMats[i], projMats[j], imgPts[i], imgPts[j])
                pos[i][j] = p4d
                # pos[j,i] = pos[i,j]

                res[i][j] = 0  # inliers
                inliers = []
                for k in range(num):
                    # reproj
                    repr = np.dot(projMats[k], p4d)
                    repr[0] /= repr[2]
                    repr[1] /= repr[2]
                    repr = repr[:2]

                    # err calc
                    diff = repr.T - imgPts[k]
                    err = np.linalg.norm(diff, 2)
                    # pprint(repr)
                    # pprint(imgPts[k])
                    # pprint(diff)
                    # print err
                    if err < err_thresh:
                        inliers.append(k)

                res[i][j] = inliers
                if best is None or len(inliers) > len(res[best[0]][best[1]]):
                    best = (i, j)

        inliers = res[best[0]][best[1]]
        if len(inliers) < min_inliers:
            return None, None

        sfmImgPoints = []
        sfmProjs = []
        for inl in inliers:
            sfmImgPoints.append(imgPts[inl])
            sfmProjs.append(projMats[inl])

        point = self.solve_sfm(sfmImgPoints, sfmProjs)
        return point, inliers

        # todo: manually add some inliers to ransac solve
        # todo: add checking of coordinate bounds to maybe class level? (eg. z is in [1, 3])
        # todo: compute pos from multiple images taken, each defining a ray to the position
        # todo: check match quality if photo taken from relatively same viewpoint
        # todo: refactor, rethink calc_data_from_files_triang

    def getCliquePosSimple(self, clique, kpts, tmats, avg_err_thresh=20, max_err_thresh = 30):
        num = len(clique)
        projMats = [np.dot(Utils.camMtx, tmat) for tmat in tmats]
        imgPts = [np.array(kpts[imgidx][0][kptidx].pt)
                  for imgidx, kptidx in clique]

        point = self.solve_sfm(imgPts, projMats)

        #reproj
        point_h = np.ones((4, 1), dtype=np.float32)
        point_h[:3,:] = point
        repr_pts = [projMat.dot(point_h).reshape(3) for projMat in projMats]
        errs = np.zeros((num,), dtype = np.float32)
        for i in range(num):
            repr_pt = repr_pts[i]
            repr_pt /= repr_pt[2]
            img_pt = imgPts[i]
            repr_pt = repr_pt[:2]
            err = np.linalg.norm(repr_pt - img_pt)
            errs[i] = err

        avg_err = np.average(errs)
        max_err = np.max(errs)
        if avg_err < avg_err_thresh and max_err < max_err_thresh:
            return point
        else:
            return None

    def getEdgePosTriangulate(self, edge, graph, kpts, tmats, max_repr_err = 20, min_num_inliers = 4):
        node1, node2 = edge
        img_idx1, kpt_idx1 = node1
        img_idx2, kpt_idx2 = node2
        impt1 = np.array(kpts[img_idx1][0][kpt_idx1].pt)
        impt2 = np.array(kpts[img_idx2][0][kpt_idx2].pt)
        projMats = [np.dot(Utils.camMtx, tmat) for tmat in tmats]
        p4d = self._triangulate(projMats[img_idx1], projMats[img_idx2], impt1, impt2)

        pt1 =  self._calc_inliers(graph, kpts, node1, p4d, projMats, max_repr_err, min_num_inliers)
        pt2 =  self._calc_inliers(graph, kpts, node2, p4d, projMats, max_repr_err, min_num_inliers)
        return pt1, pt2

    def _calc_inliers(self, graph, kpts, node, p4d, projMats, max_repr_err, min_num_inliers):
        neighbours = graph[node]
        inliers = []
        for n in neighbours:
            # reproj
            neigh_im_idx, neigh_kpt_idx = n
            repr = np.dot(projMats[neigh_im_idx], p4d)
            repr = repr[:2] / repr[2]

            # err calc
            diff = repr.T - np.array(kpts[neigh_im_idx][0][neigh_kpt_idx].pt)
            err = np.linalg.norm(diff, 2)
            # pprint(repr)
            # pprint(imgPts[k])
            # pprint(diff)
            # print err
            if err < max_repr_err:
                inliers.append(n)
        if len(inliers) >= min_num_inliers:
            sfm_imgpts = [np.array(kpts[inl_im][0][inl_kpt].pt) for inl_im, inl_kpt in inliers]
            sfm_projs = [projMats[inl_im] for inl_im, inl_kpt in inliers]
            return self.solve_sfm(sfm_imgpts, sfm_projs)
        return None

    def solve_sfm(self, img_pts, projs):
        num_imgs = len(img_pts)

        G = np.zeros((num_imgs * 2, 3), np.float64)
        b = np.zeros((num_imgs * 2, 1), np.float64)
        for i in range(num_imgs):
            ui = img_pts[i][0]
            vi = img_pts[i][1]
            p1i = projs[i][0, :3]
            p2i = projs[i][1, :3]
            p3i = projs[i][2, :3]
            a1i = projs[i][0, 3]
            a2i = projs[i][1, 3]
            a3i = projs[i][2, 3]
            idx1 = i * 2
            idx2 = idx1 + 1

            G[idx1, :] = ui * p3i - p1i
            G[idx2, :] = vi * p3i - p2i
            b[idx1] = a1i - a3i * ui
            b[idx2] = a2i - a3i * vi
        x = np.dot(np.linalg.pinv(G), b)
        return x

    # pointData is list of tuple: (des, p3d, img_idx, kpt_idx)
    def calc_data_from_files_triang(self, datafile=DC.POINTS4D_HOMOGR_TRIANG, noload=False):
        imgs = [cv2.imread(f) for f in self.filenames]
        matches, kpts = self.getMatches()

        data = None if noload else DC.getData(datafile)
        if data is None:
            graph = self.getGraph(matches, kpts)

            all_levels = self.extractCliques(graph, maxlevel=3)
            # sfm.extendCliques(graph, all_levels[0], 1)
            # all_levels = sfm.extractCliques(graph, maxlevel=3)
            # sfm.extendCliques(graph, all_levels[0], 1)
            # all_levels = sfm.extractCliques(graph, maxlevel=3)

            tmats = [MarkerDetect.loadMat(f) for f in self.filenames]
            points = []
            pointData = []

            for i in range(len(all_levels[0])):
                if i % 1000 == 0: print i, len(all_levels[0]), len(pointData)
                c = all_levels[0][i]

                edge1 = (c[0], c[1])
                edge2 = (c[1], c[2])
                edge3 = (c[0], c[2])

                edge = edge1
                pt0, pt1 = self.getEdgePosTriangulate(edge, graph, kpts, tmats)
                if pt0 is not None:
                    img_idx, kpt_idx = edge[0]
                    pointData.append((kpts[img_idx][1][kpt_idx], pt0, img_idx, kpt_idx))
                if pt1 is not None:
                    img_idx, kpt_idx = edge[1]
                    pointData.append((kpts[img_idx][1][kpt_idx], pt1, img_idx, kpt_idx))

                edge = edge2
                pt0, pt1 = self.getEdgePosTriangulate(edge, graph, kpts, tmats)
                if pt0 is not None:
                    img_idx, kpt_idx = edge[0]
                    pointData.append((kpts[img_idx][1][kpt_idx], pt0, img_idx, kpt_idx))
                if pt1 is not None:
                    img_idx, kpt_idx = edge[1]
                    pointData.append((kpts[img_idx][1][kpt_idx], pt1, img_idx, kpt_idx))

                edge = edge3
                pt0, pt1 = self.getEdgePosTriangulate(edge, graph, kpts, tmats)
                if pt0 is not None:
                    img_idx, kpt_idx = edge[0]
                    pointData.append((kpts[img_idx][1][kpt_idx], pt0, img_idx, kpt_idx))
                if pt1 is not None:
                    img_idx, kpt_idx = edge[1]
                    pointData.append((kpts[img_idx][1][kpt_idx], pt1, img_idx, kpt_idx))

            DC.saveData(datafile, (points, pointData))
        else:
            points, pointData = data

        return imgs, kpts, points, pointData

    # pointData is list of tuple: (des, p3d, img_idx, kpt_idx)
    def calc_data_from_files_triang_simple(self, noload=False):
        imgs = [cv2.imread(f) for f in self.filenames]
        matches, kpts = self.getMatches()

        import DataCache as DC
        datafile = DC.POINTS4D_HOMOGR_TRIANG_SIMPLE
        data = None if noload else DC.getData(datafile)
        if data is None:
            graph = self.getGraph(matches, kpts)

            points = []
            pointData = []
            tmats = [MarkerDetect.loadMat(f) for f in self.filenames]
            projMats = [np.dot(Utils.camMtx, tmat) for tmat in tmats]
            for node in graph:
                for neigh in graph[node]:
                    im_idx1 = node[0]
                    im_idx2 = neigh[0]
                    kpt_idx1 = node[1]
                    kpt_idx2 = neigh[1]

                    imgPt1 = np.array(kpts[im_idx1][0][kpt_idx1].pt)
                    imgPt2 = np.array(kpts[im_idx2][0][kpt_idx2].pt)

                    p4d = self._triangulate(projMats[node[0]], projMats[neigh[0]], imgPt1, imgPt2)
                    p4d = p4d[:3, :]
                    pointData.append((kpts[im_idx1][1][kpt_idx1], p4d, im_idx1, kpt_idx1))

            DC.saveData(datafile, (points, pointData))
        else:
            points, pointData = data

        return imgs, kpts, points, pointData

    # pointData is list of tuple: (des, p3d, img_idx, kpt_idx)
    def calc_data_from_files_triang_ransac(self, noload=False):
        imgs = [cv2.imread(f) for f in self.filenames]
        matches, kpts = self.getMatches()

        import DataCache as DC
        datafile = DC.POINTS4D_HOMOGR_TRIANG_RANSAC
        data = None if noload else DC.getData(datafile)
        if data is None:
            graph = self.getGraph(matches, kpts)

            points = []
            pointData = []
            tmats = [MarkerDetect.loadMat(f) for f in self.filenames]

            num_done = 0
            for node in graph:
                if num_done % 1000 == 0:
                    print num_done, len(graph.keys()), len(pointData)
                num_done += 1

                nodes = [n for n in graph[node]]
                nodes.append(node)
                point, inliers = self.getCliquePosRANSAC(nodes, kpts, tmats, err_thresh=10)

                if point is not None:
                    im_idx = node[0]
                    kpt_idx = node[1]
                    pointData.append((kpts[im_idx][1][kpt_idx], point, im_idx, kpt_idx))

            DC.saveData(datafile, (points, pointData))
        else:
            points, pointData = data

        return imgs, kpts, points, pointData

    def _triangulate(self, projMat1, projMat2, imgPts1, imgPts2):
        p4d = cv2.triangulatePoints(projMat1, projMat2, imgPts1, imgPts2)

        for i in range(0, p4d.shape[1]):
            p4d[0][i] /= p4d[3][i]
            p4d[1][i] /= p4d[3][i]
            p4d[2][i] /= p4d[3][i]
            p4d[3][i] /= p4d[3][i]

        return p4d

def draw_clique(clique, imgs, kpts):
    m = clique
    print "-- draw --"
    print m
    c = None
    for j in range(1, len(m)):
        img_idx1 = m[0][0]
        img_idx2 = m[j][0]
        kpt_idx1 = m[0][1]
        kpt_idx2 = m[j][1]
        print(img_idx1, img_idx2, kpt_idx1, kpt_idx2)

        img1 = imgs[img_idx1]
        img2 = imgs[img_idx2]

        pt1 = kpts[img_idx1][0][kpt_idx1].pt
        pt2 = kpts[img_idx2][0][kpt_idx2].pt
        Utils.drawMatch(img1, img2, pt1, pt2, scale=4)
        c = cv2.waitKey()
    return c

def calc_repr_err(c, p, inl, tmats, kpts):
    errs = [0] * len(inl)
    p4d = np.ones((4,1))
    p4d[:3,:] = p
    for i in range(len(inl)):
        inlier = inl[i]
        img_idx, kpt_idx = c[inlier]
        kpt = np.array(kpts[img_idx][0][kpt_idx].pt, dtype=np.float32)

        tmat = tmats[img_idx]
        cmat = Utils.camMtx
        reproj = cmat.dot(tmat).dot(p4d)
        reproj /= reproj[2, 0]
        errs[i] = np.linalg.norm(reproj.T[0,:2] - kpt)
        print errs[i]
    max_err = max(errs)
    avg_err = np.average(np.array(errs))
    print max_err, avg_err

def draw_real_coords(img, img_pts, obj_pts, print_coords = False, skip = 1):
    img2 = np.copy(img)
    for i in range(0, len(img_pts), skip):
        img_pt = img_pts[i]
        obj_pt = obj_pts[i]
        img_pt = tuple(map(int, img_pt))
        txt = str(np.round(obj_pt.T, 1))
        if print_coords:
            print i, txt
        cv2.circle(img2, img_pt, 7, (0, 0, 255), 3)
        cv2.putText(img2, str(i), (img_pt[0] + 5, img_pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 255, 0), 3)
    img2 = cv2.pyrDown(img2)
    cv2.imshow("",img2)
    cv2.waitKey()

def match_multiple_imgs(file1, file2, imgs, kpts, points, data):
    tmat1, tmreal1, numinl = match_to_img(file1, imgs, kpts, points, data, False, True)
    tmat2, tmreal2, numinl = match_to_img(file2, imgs, kpts, points, data)

    tmat1 = Utils.cvt_3x4_to_4x4(tmat1)
    tmat2 = Utils.cvt_3x4_to_4x4(tmat2)
    tmreal1 = Utils.cvt_3x4_to_4x4(tmreal1)
    tmreal2 = Utils.cvt_3x4_to_4x4(tmreal2)
    c2c1 = tmreal1.dot(np.linalg.inv(tmreal2))

    # o1 o2 are the origins of the camera coord systems.
    # ow1, ow2 are the estimated positions of the origin of the object (world) coord sys.
    # all coords in camera1 coord sys (so o1 is zero vector)

    o1 = np.array([[0, 0, 0, 1.0]]).T[:3,:]
    o2 = c2c1.dot(np.array([[0, 0, 0, 1.0]]).T)[:3,:]
    ow1 = tmat1.dot(np.array([[0, 0, 0, 1.0]]).T)[:3,:]
    ow2 = c2c1.dot(tmat2.dot(np.array([[0, 0, 0, 1.0]]).T))[:3,:]

    print ow1
    print ow2
    print tmreal1.dot(np.array([[0, 0, 0, 1.0]]).T)
    print calc_midpoint(o1, o2, ow1-o1, ow2-o2)

def calc_midpoint(p1, p2, v1, v2):
    v1_ = v1.reshape((3,))
    v2_ = v2.reshape((3,))
    n = np.cross(v1_, v2_).reshape((3,1))
    A = np.array([v2, -v1, n]).reshape((3,3)).T
    b = p1 - p2
    x = np.linalg.inv(A).dot(b)
    u, t, lbd = x[0,0], x[1,0], x[2,0]
    mp = (t * v1 + p1 + u * v2 + p2) / 2
    return mp

def match_to_img(file, imgs, kpts, points, data, draw_coords = True, draw_inl = False, repr_err_thresh = 20, draw_outliers = True):
    print "match_to_img:  %s" % (file)
    img = cv2.imread(file)
    data_des, data_pts, img_indices, kpt_indices = zip(*data)

    fl = FeatureLoader.FeatureLoader()
    kp, des = fl.loadFeatures(file, "surf")
    ml = MatchLoader.MatchLoader()

    while len(data_des) > 2 ** 18:
        data_des = tuple([data_des[i] for i in range(0, len(data_des), 2)])
    print len(data_des)

    matches = ml.matchBFCross(file, "asd/nope.avi", des, data_des, "surf", nosave=True, noload=True)
    print len(matches)

    # dct = {}
    # for i in range(48):
    #     dct[i] = 0
    # for m in matches:
    #     dct[img_indices[m.trainIdx]] += 1
    # print  dct

    img_pts = []
    obj_pts = []
    for m in matches:
        img_pts.append(kp[m.queryIdx].pt)
        obj_pts.append(data_pts[m.trainIdx])

    print "repr_err = %d" % repr_err_thresh
    rvec, tvec, inliers = cv2.solvePnPRansac(
        np.asarray(obj_pts, np.float32),
        np.asarray(img_pts, np.float32),
        Utils.camMtx,
        None,
        iterationsCount=100,
        reprojectionError=repr_err_thresh)

    if draw_inl:
        for i in range(len(matches)):
            is_inlier = i in inliers
            m = matches[i]
            data_idx = m.trainIdx
            imidx, kptidx = img_indices[data_idx], kpt_indices[data_idx]
            img1 = imgs[imidx]
            kpt1 = kpts[imidx][0][kptidx].pt
            img2 = img
            kpt2 = img_pts[i]
            if is_inlier or draw_outliers:
                Utils.drawMatch(img1, img2, kpt1, kpt2, is_inlier, 2)
                print i
                c = cv2.waitKey()
                if c == 27:
                    break


    img_pts2 = [img_pts[i] for i in range(len(img_pts)) if i in inliers]
    obj_pts2 = [obj_pts[i] for i in range(len(obj_pts)) if i in inliers]
    if draw_coords:
        draw_real_coords(img, img_pts2, obj_pts2, True, 5)

    rmat = cv2.Rodrigues(rvec)[0]

    tmat_real = MarkerDetect.loadMat(file)
    tmat = np.zeros((3,4))
    tmat[:3,:3] = rmat
    tmat[:3,3] = tvec.T
    tmat4x4inv = Utils.invTrf(tmat)
    print "num data, img pts", len(data_des), len(des)
    print "num matches:", len(matches)
    print "num inliers: ", len(inliers)
    print "rmat", rmat
    print "tvec", tvec
    if tmat_real is not None:
        print "tmat load", tmat_real
        print "combined trf diff", tmat_real.dot(tmat4x4inv)
        world_origin = np.array([0, 0, 0, 1], dtype=np.float32).T
        world_origin = tmat_real.dot(world_origin)
        est_origin = np.array([0, 0, 0, 1], dtype=np.float32).T
        est_origin = tmat.dot(est_origin)
        print "real orig", world_origin
        print "est orig", est_origin
    else:
        est_origin = np.array([0, 0, 0, 1], dtype=np.float32).T
        est_origin = tmat.dot(est_origin)
        print "est orig", est_origin
    return tmat, tmat_real, len(inliers)

def test(file):
    from glob import glob
    from os.path import  join
    files_dir = "out/2017_3_8__14_51_22/"
    files = glob(join(files_dir, "*.jpg"))
    # files = [f for f in files if f != file]
    # print files
    masks = []
    for f in files:
        m = f.replace(".jpg", "_mask.png")
        masks.append(m)
    sfm = SFMSolver(files, masks)

    imgs, kpts, points, data = sfm.calc_data_from_files_triang_simple()



    print "len pointData %d" % len(data)

    match_to_img(file, imgs, kpts, points, data, False, repr_err_thresh=20)
    return

    print "num points: ", len(points)
    for c, p, a, m in points:
        print "--- new clique ---"
        print p
        print "error (avg, max): ", a, m
        if p[2] > -1.5:
            if draw_clique(c, imgs, kpts) == 27:
                return

def ransac_test():
    from glob import glob
    from os.path import join
    class foo:
        def __init__(self, pt_):
            self.pt = pt_
    files_dir = "out/2017_3_8__14_51_22/"
    files = glob(join(files_dir, "*.jpg"))
    masks = []
    for f in files:
        m = f.replace(".jpg", "_mask.png")
        masks.append(m)

    sfm = SFMSolver(files, masks)

    numpts = 10
    numout = 2
    pt = np.array([1, 10, 3, 1], dtype="float64").T
    tmats = []
    kpts = []
    nodes = []
    from random import random
    for i in range(numpts):
        tmat = Utils.getTransform(
            random() * 2 - 1,
            random() * 2 - 1,
            random() * 2 - 1,
            random() * 30 + 30,
            random() * 30 + 30,
            random() * 30 + 30)
        tmats.append(tmat)
        projpt = Utils.camMtx.dot(tmat.dot(pt))
        projpt /= projpt[2]
        projpt = projpt[:2]
        if i < numout:
            projpt += np.random.rand(2,) * 100

        kpts.append(([foo(projpt)], None))
        nodes.append((i, 0))

    print sfm.getCliquePosRANSAC(nodes, kpts, tmats, err_thresh=20)

def test_two_lines():
    files = ["imgs/00%d.jpg" % (i) for i in range(5, 10)]
    imgs, kpts, points, data = calc_data_from_files(files)
    match_multiple_imgs("imgs/003.jpg", "imgs/004.jpg", imgs, kpts, points, data)
    exit()

# pointData is list of tuple: (des, p3d, img_idx, kpt_idx)
def calc_data_from_files(files, noload = False, datafile = DC.POINTS4D):
    imgs = [cv2.imread(f) for f in files]
    masks = [cv2.imread("imgs/00%d_mask.png" % i, 0) for i in range(5, 10)]
    sfm = SFMSolver(files, masks)
    matches, kpts = sfm.getMatches()

    import DataCache as DC
    data = None if noload else DC.getData(datafile)
    if data is None:
        graph = sfm.getGraph(matches, kpts)
        all_levels = sfm.extractCliques(graph, maxlevel=3)
        # sfm.extendCliques(graph, all_levels[0], 1)
        # all_levels = sfm.extractCliques(graph, maxlevel=3)
        # sfm.extendCliques(graph, all_levels[0], 1)
        # all_levels = sfm.extractCliques(graph, maxlevel=3)

        tmats = [MarkerDetect.loadMat(f) for f in files]
        points = []

        # for c in all_levels[0]:
        #     point, inliers = sfm.getCliquePosRANSAC(c, kpts, tmats, err_thresh=100)
        #     if point is not None:
        #         points.append((c, point, inliers))
        # print "num points: ", len(points)
        # for c, p, inl in points:
        #     print "--- new clique ---"
        #     # print p
        #     # calc_repr_err(c, p, inl, tmats, kpts)
        #     if draw(c, imgs, kpts) == 27:
        #         return

        for i in range(len(all_levels[0])):
            if i % 1000 == 0: print i, len(all_levels[0]), len(points)
            c = all_levels[0][i]
            point = sfm.getCliquePosSimple(c, kpts, tmats, avg_err_thresh=5, max_err_thresh=10)
            if point is not None:
                points.append((c, point))

        pointData = []
        for c, p in points:
            for node in c:
                img_idx, kpt_idx = node
                pointData.append((kpts[img_idx][1][kpt_idx], p, img_idx, kpt_idx))

        DC.saveData(datafile, (points, pointData))
    else:
        points, pointData = data

    return imgs, kpts, points, pointData

def calc_data_from_files_unif(files, noload = False, datafile = DC.POINTS4D_UNIFIED):
    imgs = [cv2.imread(f) for f in files]
    masks = [cv2.imread("imgs/00%d_mask.png" % i, 0) for i in range(5, 10)]
    sfm = SFMSolver(files, masks)
    matches, kpts = sfm.getMatches()

    data = None if noload else DC.getData(datafile)
    assert data is not None
    points, pointData = data
    return imgs, kpts, points, pointData

# pointData is list of tuple: (des, p3d, img_idx, kpt_idx)
def calc_data_from_files_triang(files, datafile = DC.POINTS4D_TRIANGULATE, noload = False):
    imgs = [cv2.imread(f) for f in files]
    masks = [cv2.imread("imgs/00%d_mask.png" % i, 0) for i in range(5, 10)]
    sfm = SFMSolver(files, masks)
    matches, kpts = sfm.getMatches()

    data = None if noload else DC.getData(datafile)
    if data is None:
        graph = sfm.getGraph(matches, kpts)
        all_levels = sfm.extractCliques(graph, maxlevel=3)
        # sfm.extendCliques(graph, all_levels[0], 1)
        # all_levels = sfm.extractCliques(graph, maxlevel=3)
        # sfm.extendCliques(graph, all_levels[0], 1)
        # all_levels = sfm.extractCliques(graph, maxlevel=3)

        tmats = [MarkerDetect.loadMat(f) for f in files]
        points = []
        pointData = []

        for i in range(len(all_levels[0])):
            if i % 1000 == 0: print i, len(all_levels[0]), len(pointData)
            c = all_levels[0][i]

            edge1 = (c[0], c[1])
            edge2 = (c[1], c[2])
            edge3 = (c[0], c[2])

            edge = edge1
            pt0, pt1 = sfm.getEdgePosTriangulate(edge, graph, kpts, tmats)
            if pt0 is not None:
                img_idx, kpt_idx = edge[0]
                pointData.append((kpts[img_idx][1][kpt_idx], pt0, img_idx, kpt_idx))
            if pt1 is not None:
                img_idx, kpt_idx = edge[1]
                pointData.append((kpts[img_idx][1][kpt_idx], pt1, img_idx, kpt_idx))

            edge = edge2
            pt0, pt1 = sfm.getEdgePosTriangulate(edge, graph, kpts, tmats)
            if pt0 is not None:
                img_idx, kpt_idx = edge[0]
                pointData.append((kpts[img_idx][1][kpt_idx], pt0, img_idx, kpt_idx))
            if pt1 is not None:
                img_idx, kpt_idx = edge[1]
                pointData.append((kpts[img_idx][1][kpt_idx], pt1, img_idx, kpt_idx))

            edge = edge3
            pt0, pt1 = sfm.getEdgePosTriangulate(edge, graph, kpts, tmats)
            if pt0 is not None:
                img_idx, kpt_idx = edge[0]
                pointData.append((kpts[img_idx][1][kpt_idx], pt0, img_idx, kpt_idx))
            if pt1 is not None:
                img_idx, kpt_idx = edge[1]
                pointData.append((kpts[img_idx][1][kpt_idx], pt1, img_idx, kpt_idx))

        DC.saveData(datafile, (points, pointData))
    else:
        points, pointData = data

    return imgs, kpts, points, pointData

# pointData is list of tuple: (des, p3d, img_idx, kpt_idx)
def calc_data_from_files_triang_simple(files, noload = False):
    imgs = [cv2.imread(f) for f in files]
    masks = [cv2.imread("imgs/00%d_mask.png" % i, 0) for i in range(5, 10)]
    sfm = SFMSolver(files, masks)
    matches, kpts = sfm.getMatches()

    import DataCache as DC
    datafile = DC.POINTS4D_HOMOGR_TRIANG_SIMPLE
    data = None if noload else DC.getData(datafile)
    if data is None:
        graph = sfm.getGraph(matches, kpts)


        points = []
        pointData = []
        tmats = [MarkerDetect.loadMat(f) for f in files]
        projMats = [np.dot(Utils.camMtx, tmat) for tmat in tmats]
        for node in graph:
            for neigh in graph[node]:
                im_idx1 = node[0]
                im_idx2 = neigh[0]
                kpt_idx1 = node[1]
                kpt_idx2 = neigh[1]

                imgPt1 = np.array(kpts[im_idx1][0][kpt_idx1].pt)
                imgPt2 = np.array(kpts[im_idx2][0][kpt_idx2].pt)

                p4d = sfm._triangulate(projMats[node[0]], projMats[neigh[0]], imgPt1, imgPt2)
                p4d = p4d[:3,:]
                pointData.append((kpts[im_idx1][1][kpt_idx1], p4d, im_idx1, kpt_idx1))

        DC.saveData(datafile, (points, pointData))
    else:
        points, pointData = data

    return imgs, kpts, points, pointData

def projtest():
    numpts = 10
    tro = Utils.getTransform(0, 0, 0, 330, 0, 0, True)
    trt = Utils.getTransform(180, 0, 180, 300, 0, 500, True)
    ttc = Utils.getTransform(0, 0, 0, 0, 0, 20, True)
    tco = np.linalg.inv(ttc).dot(np.linalg.inv(trt).dot(tro))
    # print tro
    # print np.dot(trt, ttc.dot(tco))
    objpts = np.zeros((numpts, 4))
    objpts[:,:2] = np.random.rand(numpts, 2) * 10
    objpts[:,3] = np.ones((numpts,))
    objpts = objpts.T
    print objpts
    imgpts = Utils.camMtx.dot(tco[:3,:].dot(objpts))
    for i in range(numpts):
        imgpts[:,i] /= imgpts[2,i]
    print imgpts
    objpts = objpts[:3, :].T.reshape(-1, 1, 3)
    imgpts = imgpts[:2, :].T.reshape(-1, 1, 2)

    imgpts_ = imgpts + (np.random.rand(numpts, 1, 2) * 2 - np.ones_like(imgpts)) * 0
    print imgpts - imgpts_

    _, rvec, tvec = cv2.solvePnP(np.array(objpts), np.array(imgpts_), Utils.camMtx, None)
    print "--"

    tco_est = np.eye(4)
    tco_est[:3,:3] = cv2.Rodrigues(rvec)[0]
    tco_est[:3,3] = tvec.reshape((3, ))

    tro_est = trt.dot(ttc.dot(tco_est))
    print tro-tro_est

def findtest():
    from glob import glob
    from os.path import join

    np.set_printoptions(precision=3, suppress=True)

    files_dir = "out/2017_3_8__14_51_22/"
    files = glob(join(files_dir, "*.jpg"))
    # files = [f for f in files if f != file]
    # print files
    masks = []
    for f in files:
        m = f.replace(".jpg", "_mask.png")
        masks.append(m)
    sfm = SFMSolver(files, masks)
    imgs, kpts, points, data = sfm.calc_data_from_files_triang_simple()
    arr_calib = DC.getData("out/2017_4_5__15_6_49/arrangement_calib.p")
    ttc = arr_calib["ttc"]
    tor = arr_calib["tor"]
    files_dir = "out/2017_3_8__14_51_22/"
    files_dir = "out/2017_4_5__15_31_34/"
    files_dir = "out/2017_4_5__15_57_20/"

    files = glob(join(files_dir, "*.jpg"))
    files = glob(join(files_dir, "0037.jpg"))

    for f in files:
        find_ext_params(f, imgs, kpts, points, data, tor, ttc)


def find_ext_params(filename, imgs, kpts, points, data, tor, ttc, draw_coords = False, draw_inl = False, draw_outliers = True):
    print filename
    tco_est, tco_real, numinl = match_to_img(filename, imgs, kpts, points, data, draw_coords=draw_coords, draw_inl=draw_inl, repr_err_thresh=10, draw_outliers=draw_outliers)
    cba_goal, xyz_goal = estimate(filename, tco_est, tco_real, tor, ttc)
    return xyz_goal[:3], cba_goal, numinl


def estimate(filename, tco_est, tco_real, tor, ttc):
    temp = np.eye(4)
    temp[:3, :] = tco_est
    tco_est = temp
    temp = np.eye(4)
    temp[:3, :] = tco_real
    tco_real = temp
    pos_data_file = filename.replace("jpg", "p")
    posdata = DC.getData(pos_data_file)
    x, y, z, a, b, c = [posdata[0][i] for i in [500, 501, 502, 503, 504, 505]]
    a, b, c = map(lambda p: p * np.pi / 180, (a, b, c))  # deg to rad
    x, y, z = map(lambda p: p / 10.0, (x, y, z))  # mm to cm
    # print x, y, z,'|', a, b, c
    # img = cv2.imread(f)
    # img = cv2.pyrDown(img)
    # cv2.imshow("asd", img)
    # cv2.waitKey()
    trt = Utils.getTransform(c, b, a, x, y, z, True)
    # print trt.shape, ttc.shape, tco_est.shape
    print "---"
    print x, y, z, a, b, c
    print trt
    print ttc
    print tco_est
    # print tco_real
    print "---"
    # print np.linalg.inv(tco_real.dot(tor.dot(trt))) #ttc
    tro_est = trt.dot(ttc.dot(tco_est))
    print "tro est"
    print tro_est
    print "tro real"
    print np.linalg.inv(tor)
    print "cam pos est"
    pos_goal = [11, 5.2, -4, 1.0]
    # pos_goal = [0, 0, 0, 1.0]
    xyz_goal = tro_est.dot(np.array(pos_goal).T)
    print xyz_goal  # becsult pozicioja a kameranak amikor kozel volt
    # rr, pp, yy = map(lambda v: v * np.pi / 180, (-180, -14, -180))
    # print "rpy trf real"
    # print Utils.getTransform(rr, pp, yy, 0, 0, 0)
    print "trt_goal \n a, b, c "
    toc_goal = np.array([-1, 0, 0, pos_goal[0], 0, -1, 0, pos_goal[1], 0, 0, 1, pos_goal[2], 0, 0, 0, 1]).reshape(
        (4, 4))
    # toc_goal = np.array([1, 0, 0, pos_goal[0], 0, 1, 0, pos_goal[1], 0, 0, 1, pos_goal[2], 0, 0, 0, 1]).reshape(
    #     (4, 4))
    trt_goal = tro_est.dot(toc_goal.dot(np.linalg.inv(ttc)))
    cba_goal = np.array(map(np.rad2deg, Utils.rpy(trt_goal[:3, :3])))
    print trt_goal
    print cba_goal
    # print Utils.getTransform(rr, pp, yy, 0, 0, 0)
    return cba_goal, xyz_goal


def estimate_test():
    np.set_printoptions(3, suppress=True)
    filename = "out/test/test.p"
    tro = Utils.getTransform(1, 2, 3, 4, 5, 6, True)
    tor = np.linalg.inv(tro)
    x, y, z = 1, 2, 3
    da, db, dc = 10, 20, 30
    ra, rb, rc = map(np.deg2rad, (da, db, dc))
    trt = Utils.getTransform(rc, rb, ra, x, y, z, True)
    ttr = np.linalg.inv(trt)
    ttc = Utils.getTransform(2, 3, 5, 4, 6, 1, True)
    tct = np.linalg.inv(ttc)
    tco = tct.dot(ttr).dot(tro)
    toc = np.linalg.inv(tco)
    xyz_g_ex = np.array([11, 5.2, -4, 1.0]).reshape((4, 1)) #objectben
    xyz_g_ex = tro.dot(xyz_g_ex)
    toc_goal = np.array([-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).reshape((4, 4))
    rot_g_ex = tro.dot(toc_goal).dot(tct)[:3, :3]
    cba_g_ex = map(np.rad2deg, Utils.rpy(rot_g_ex))
    print trt
    print ttc
    print tco
    print trt.dot(ttc.dot(tco))

    data = [{
        500: x * 10,
        501: y * 10,
        502: z * 10,
        503: da,
        504: db,
        505: dc
    }]
    DC.saveData(filename, data)
    print "estimate call"
    cba_g, xyz_g = estimate(filename.replace(".p", ".jpg"), tco[:3, :], tco[:3, :], tor, ttc)
    print "estimate return"
    print "----"
    print cba_g, xyz_g
    print "exact: "
    print cba_g_ex
    print xyz_g_ex
    trf_orig = Utils.getTransform(rc, rb, ra, x, y, z)
    print trf_orig
    rc, rb, ra = np.deg2rad(cba_g)
    x, y, z = xyz_g[:3]
    trf_est = Utils.getTransform(rc, rb, ra, x, y, z)
    print trf_est
    print np.max(np.abs(trf_est-trf_orig))



if __name__ == '__main__':
    # findtest() #<---- EREDMENYEK!!!!!!!!!!!!!!!

    estimate_test()

    # ransac_test()
    exit()

    # import cProfile
    # cProfile.run("test()")
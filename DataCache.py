import cPickle as pickle
from os.path import isfile

POINTS4D = "cache/points4d.p" # epilines + extension x2 + getCliquePosSimple
POINTS4D_MULTIPLE_MATCH = "cache/points4d_multiple.p" # epilines_multiple + getCliquePosSimple
POINTS4D_TRIANGULATE = "cache/points4d_triangulate.p" # epilines + triangulate
POINTS4D_UNIFIED = "cache/points4d_unified.p"   # POINTS4D_TRIANGULATE + POINTS4D
POINTS4D_HOMOGR_TRIANG_SIMPLE = "cache/points4d_homogr_triang_simple.p"   # epilines homogr + triang simple
POINTS4D_HOMOGR_TRIANG = "cache/points4d_homogr_triang.p"   # epilines homogr + triang
POINTS4D_HOMOGR_TRIANG_RANSAC = "cache/points4d_homogr_triang_ransac.p"   # epilines homogr + triang ransac

# DO NOT USE
POINTS4D_UNIFIED_ALL = "cache/points4d_unified_all.p"   # POINTS4D_TRIANGULATE + POINTS4D + POINTS4D_MULTIPLE_MATCH

def getData(filename):
    print "DataCache get: %s" % filename
    if isfile(filename):
        f = open(filename, "rb")
        data = pickle.load(f)
        f.close()
        return data
    return None

def saveData(filename, data):
    f = open(filename, "wb")
    pickle.dump(data, f, 2)
    f.close()

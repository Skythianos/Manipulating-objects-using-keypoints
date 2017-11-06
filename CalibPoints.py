from pprint import pprint
import math
import numpy as np
import Utils

ORIGIN = [300, 0, 500, 180, 0, 180] #xyzabc, mm and degrees

def interp(xmin, xmax, ratio):
    return xmin * (1.0 - ratio) + 1.0 * ratio * xmax

def points1_gen():
    """
    generates 11 points along all 3 axes in the range [-5, 5]
    """

    points = []
    for i in range(-5, 6):
        points.append([i * 10, 0, 0, 0, 0, 0])
    for i in range(-5, 6):
        points.append([0, i * 10, 0, 0, 0, 0])
    for i in range(-5, 6):
        points.append([0, 0, i * 10, 0, 0, 0])

    points = [[p[i] + ORIGIN[i] for i in range(len(ORIGIN))] for p in points]
    return points, len(points)

def points2_gen():
    """
    generates 11 points along all 3 axes in the range [-5, 5],
    and 10 points rotated along both horizontal ayes
    """
    points, num = points1_gen()
    num = 0

    est_height = ORIGIN[2]
    angle_range_x = (-9, 30)
    angle_range_y = (-15, 15)
    num_points_in_dir = 10

    points2 = []
    points2.append([0, 0, 0, 0, 0, 0])
    #x
    for i in range(num_points_in_dir + 1):
        angle_deg = int(interp(angle_range_x[0], angle_range_x[1], 1.0 * i / num_points_in_dir))
        angle_rad = math.radians(angle_deg)
        delta = int(math.atan(angle_rad) * est_height)
        points2.append([-delta, 0, 0, 0, angle_deg, 0])

    #y
    for i in range(num_points_in_dir + 1):
        angle_deg = int(interp(angle_range_y[0], angle_range_y[1], 1.0 * i / num_points_in_dir))
        angle_rad = math.radians(angle_deg)
        delta = int(math.atan(angle_rad) * est_height)
        points2.append([0, delta, 0, angle_deg, 0, 0])

    points2 = [[p[i] + ORIGIN[i] for i in range(len(ORIGIN))] for p in points2]
    return points + points2, num

def find_points_gen():
    est_height = ORIGIN[2]
    points = []
    points.append([0, 0, 0, 0, 0, 0])

    angle_deg = -9
    angle_rad = math.radians(angle_deg)
    delta = int(math.atan(angle_rad) * est_height)
    points.append([-delta, 0, 0, 0, angle_deg, 0])

    angle_deg = 30
    angle_rad = math.radians(angle_deg)
    delta = int(math.atan(angle_rad) * est_height)
    points.append([-delta, 0, 0, 0, angle_deg, 0])

    angle_deg = 15
    angle_rad = math.radians(angle_deg)
    delta = int(math.atan(angle_rad) * est_height)
    points.append([0, delta, 0, angle_deg, 0, 0])

    angle_deg = -15
    angle_rad = math.radians(angle_deg)
    delta = int(math.atan(angle_rad) * est_height)
    points.append([0, delta, 0, angle_deg, 0, 0])
    points = [[p[i] + ORIGIN[i] for i in range(len(ORIGIN))] for p in points]

    return points

def points3_gen():
    radius = 500
    target = [300, 0, 0]
    angle_range_x = (-30, 30)
    angle_range_y = (-10, 30)
    resolution = (11, 11)
    # resolution = (2, 2)

    points = []
    target_vec = np.array(target).reshape((3, 1))
    over_target_vec = np.array([0, 0, radius]).reshape((3, 1))
    # x
    for i in range(resolution[0]):
        angle_deg_x = int(interp(angle_range_x[0], angle_range_x[1], 1.0 * i / (resolution[0] - 1)))
        angle_rad_x = math.radians(angle_deg_x)

        # y
        for j in range(resolution[1]):
            angle_deg_y = int(interp(angle_range_y[0], angle_range_y[1], 1.0 * j / (resolution[1] - 1)))
            angle_rad_y = math.radians(angle_deg_y)

            tool = Utils.getTransform(0, angle_rad_y, angle_rad_x, 0, 0, 0)[:3, :3].dot(over_target_vec) + target_vec
            points.append([
                int(tool[0]),
                int(tool[1]),
                int(tool[2]),
                180 - angle_deg_x,
                -angle_deg_y,
                180])
            # print angle_deg_x, angle_deg_y
            # trf = points[-1]
            # x, y, z, a, b, c = trf
            # a, b, c = map(math.radians, (a, b, c))
            # print Utils.getTransform(c, b, a, x, y, z)
    return points, 0

def points4_gen():
    homepos = [-220, -15, 310, 12, 90, -168]
    safepos_left = [-390, -15, 310, 12, 90, -168]
    safepos_right = [-390, 15, 310, 12, 90, -168]

    x_range = (-390, -220)
    y_range = (-80, 45)
    z_range = (240, 360)

    x_res = 11
    y_res = x_res
    z_res = x_res

    p4 = []
    for i in range(x_res):
        ratio = i / (x_res - 1.0)
        point = list(homepos)
        point[0] = int(interp(x_range[0], x_range[1], ratio))
        p4.append(point)
    for i in range(y_res):
        ratio = i / (y_res - 1.0)
        point = list(homepos)
        point[1] = int(interp(y_range[0], y_range[1], ratio))
        p4.append(point)
    for i in range(z_res):
        ratio = i / (z_res - 1.0)
        point = list(homepos)
        point[2] = int(interp(z_range[0], z_range[1], ratio))
        p4.append(point)

    p4_2 = points3_gen()
    p4_2 = trf_points(p4_2, Utils.getTransform(0, 0, 0, -300, 0, -500, True))
    p4_2 = trf_points(p4_2, Utils.getTransform(0, np.pi / 2, 0, 0, 0, 0, True))
    p4_2 = trf_points(p4_2, Utils.getTransform(0, 0, np.pi, 0, 0, 0, True))
    p4_2 = trf_points(p4_2, Utils.getTransform(0, 0, 0, -250, 0, 300, True))

    #safety features
    all_points = p4 + p4_2[0]
    idx = 1
    while idx < len(all_points):
        p0 = all_points[idx-1]
        p1 = all_points[idx]
        if p0[1] < 0 and p1[1] >= 0:
            all_points.insert(idx, safepos_left)
            all_points.insert(idx + 1, safepos_right)
            idx += 2
        elif p0[1] >= 0 and p1[1] < 0:
            all_points.insert(idx, safepos_right)
            all_points.insert(idx + 1, safepos_left)
            idx += 2
        else:
            idx += 1

    return all_points, len(p4)

def trf_points(points, trf):
    num = points[1]
    points = points[0]
    new_points = []
    for point in points:
        x, y, z, a, b, c = point
        a, b, c = map(math.radians, (a, b, c))
        point_trf = Utils.getTransform(c, b, a, x, y, z, True)
        new_trf = trf.dot(point_trf)
        c, b, a = Utils.rpy(new_trf[:3, :3])
        x = new_trf[0, 3]
        y = new_trf[1, 3]
        z = new_trf[2, 3]
        a, b, c = map(math.degrees, (a, b, c))
        x, y, z, a, b, c = map(int, (x, y, z, a, b, c))
        new_points.append([x, y, z, a, b, c])
    return new_points, num

np.set_printoptions(precision=3, suppress=True)
points1 = points1_gen()
points2 = points2_gen()
points3 = points3_gen()
points3_2 = points3_gen()
points3_2 = trf_points(points3_2, Utils.getTransform(0, 0, 0, -300, 0, -500, True))
points3_2 = trf_points(points3_2, Utils.getTransform(0, np.pi / 2, 0, 0, 0, 0, True))
points3_2 = trf_points(points3_2, Utils.getTransform(0, 0, np.pi, 0, 0, 0, True))
points3_2 = trf_points(points3_2, Utils.getTransform(0, 0, 0, -250, 0, 300, True))
points4 = points4_gen()
find_points = find_points_gen()

def test():
    # for p in points3[0]:
    #     x, y, z, a, b, c = p
    #     a, b, c = map(math.radians, (a, b, c))
    #     print p
    #     print Utils.getTransform(c, b, a, x, y, z)

    # points = points1
    # points = trf_points(points, Utils.getTransform(0, 0, 0, -300, 0, -500, True))
    # points = trf_points(points, Utils.getTransform(0, np.pi/2, 0, 0, 0, 0, True))
    # points = trf_points(points, Utils.getTransform(0, 0, 0, 300, 0, 500, True))
    # print points1
    # print points

    pprint(points2)
    pprint(find_points)

if __name__ == '__main__':

    test()
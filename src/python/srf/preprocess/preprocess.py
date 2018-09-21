import numpy as np

def compute_sigma2_factor(lors: np.ndarray):
    """
    compute the sigma2 factors of tor kernel 
    """
    p_start = lors[:, 0:3]
    p_end = lors[:, 3:6]
    x_start = p_start[:, 0]
    x_end = p_end[:, 0]
    y_start = p_start[:, 1]
    y_end = p_end[:, 1]
    
    p_diff = p_end - p_start
    # print("p_diff:\n", p_diff)
    p_dis2 = np.sum(np.square(p_diff), 1)
    # print("p_dis2.shape", p_dis2.shape)
    dsq = p_diff[:, 0]**2 + p_diff[:, 1]**2
    Rsq = x_start**2+x_end **2 + y_start**2 + y_end**2
    # print(dsq.shape)
    # print(Rsq.shape)    
    sigma2_factor = dsq**2/Rsq/p_dis2/2
    # print("lors shape:", lors.shape)
    # print("sigma2_factor shape:",sigma2_factor.shape)
    lors = np.hstack([lors, sigma2_factor.reshape([-1, 1])])
    # print("new lors shape", lors.shape)
    return lors


def partition_lors(lors: np.ndarray):
    """
    patition the input lors into three np.array
    according to the dominant direction.
    """
    dim_size1 = (lors.shape)[1]
    lors.reshape((-1, dim_size1))
    # lors = np.reshape(lors, (-1, dim_size1))
    p1 = lors[:, 0:3]
    p2 = lors[:, 3:6]

    # find the index of x_main, y_main and z_main
    diff = np.abs(p2 - p1)
    x_d = diff[:, 0]
    y_d = diff[:, 1]
    z_d = diff[:, 2]
    # x_mask = np.logical_and(x_d >= y_d,x_d >= z_d)
    # y_mask = np.logical_and(np.logical_and(x_d < y_d,y_d >= z_d), np.logical_not(x_mask))
    # z_mask = np.logical_not(np.logical_or(x_mask, y_mask))
    # xlors = lors[x_mask, :]
    # ylors = lors[y_mask, :]
    # zlors = lors[z_mask, :]
    xlors = lors[np.array([np.intersect1d(np.where(x_d >= y_d),
                                          np.where(x_d >= z_d))])].reshape((-1, dim_size1))
    ylors = lors[np.array([np.intersect1d(np.where(y_d > x_d),
                                          np.where(y_d >= z_d))])].reshape((-1, dim_size1))
    zlors = lors[np.array([np.intersect1d(np.where(z_d > x_d),
                                          np.where(z_d > y_d))])].reshape((-1, dim_size1))
    return xlors, ylors, zlors

# exchange the start point and end point if the


def swap_points(lors: np.ndarray, axis: int):
    """
    lors is an 2D array whose shape is [n, 6]
    Two points are indicated 
    in the order of (x1, y1, z1, x2, y2, z2).
    This function garantees the point2 is 
    large than the point1 in main axis(int (0, 1, 2) for (x, y, z))
    """
    dim_size1 = (lors.shape)[1]
    if axis > 2 or axis < 0:
        raise ValueError
    point1 = lors[:, 0:3]
    point2 = lors[:, 3:6]
    # print("point1")
    # print(point1)
    # print('point2')
    # print(point2)
    d = point2[:, axis] - point1[:, axis]
    # print("diff....")
    # print(np.array([np.where(d < 0)]))
    positive = lors[np.array([np.where(d >= 0)])].reshape((-1, dim_size1))
    negative = lors[np.array([np.where(d < 0)])].reshape((-1, dim_size1))
    # print("negative....")
    # print(negative)
    # print("positive")
    # print(positive)
    if dim_size1 == 7:
        swaped_nega = np.hstack((np.array(negative[:, 3:6]),
                                 np.array(negative[:, 0:3]),
                                 np.array(negative[:, 6:7])
                                 ))
        return np.vstack((positive, swaped_nega))
    elif dim_size1 > 7:
        swaped_nega = np.hstack((np.array(negative[:, 3:6]),
                                 np.array(negative[:, 0:3]),
                                 0 - np.array(negative[:, 6:7]),
                                 np.array(negative[:, 7:])
                                 ))
        return np.vstack((positive, swaped_nega))

    # negative[:,0:3], negative[:, 3:6] = negative[:,3:6], negative[:, 0:3]
    # swaped_n = np.hstack[]
    # swaped_lors = (positive, negative)
    # return swaped_lors


def cut_lors(lors: np.ndarray, limit: np.float32):
    """
    form of an individul lor: (x1, y1, z1, x2, y2, z2, t_dis)
    cut the lors according to the tof information.
    this operation reduce the lor length due to the tof kernel
    return a lor: (x1, y1, z1, x2, y2, z2, xc, yc, zc)
    """
    p_start = lors[:, 0:3]
    p_end = lors[:, 3:6]
    # print(p_end)
    p_diff = p_end - p_start
    # print("p_diff:\n", p_diff)
    p_dis = np.sqrt(np.sum(np.square(p_diff), 1))
    # print("p_dis:\n", p_dis.shape)
    dcos = np.array([p_diff[:, 0]/p_dis[:],
                     p_diff[:, 1]/p_dis[:],
                     p_diff[:, 2]/p_dis[:]]).transpose()
    # print("d_cos:\n", dcos)
    t_dis = lors[:, 6]
    # print("t_dis:\n", t_dis.shape)
    ratio = np.array(0.5 - (t_dis/p_dis))
    # print("ratio: \n", ratio)
    lor_center = np.array([ratio*p_diff[:, 0],
                           ratio*p_diff[:, 1],
                           ratio*p_diff[:, 2]]).transpose() + p_start

    # print("lor_center:\n", lor_center)

    # cut the lors
    dcs = np.sqrt(np.sum(np.square(lor_center - p_start), 1)).reshape((-1))

    # print("dcs:\n", dcs)

    index = dcs > limit

    # print("l_index:\n", index)

    # print("p_start:\n", p_start[index])

    p_start[index] = (lor_center[index] - np.array(limit *
                                                   dcos[index, :]).reshape((-1, 3)))

    # print("p_start:\n", p_start[index])
    dce = np.sqrt(np.sum(np.square(lor_center - p_end), 1)).reshape((-1))

    # print("dce:\n", dce)
    index = np.array([np.where(dce > limit)]).reshape((-1))

    # print("r_index:\n", index)

    # print("p_end:\n", p_end[index])
    p_end[index] = (lor_center[index] + np.array(limit *
                                                 dcos[index, :]).reshape((-1, 3)))
    # print("p_end:\n", p_end[index])
    return np.hstack((np.array(p_start),
                      np.array(p_end),
                      np.array(lor_center),
                      np.array(lors[:, 7:])))


def preprocess(lors: np.ndarray):
    from srf.preprocess.function.on_tor_lors import Axis
    lors = compute_sigma2_factor(lors)
    # print("corrected lor shape:", lors.shape)
    xlors, ylors, zlors = partition_lors(lors)
    xlors = swap_points(xlors, 0)
    ylors = swap_points(ylors, 1)
    zlors = swap_points(zlors, 2)
    return {Axis.x:xlors, Axis.y:ylors, Axis.z:zlors}

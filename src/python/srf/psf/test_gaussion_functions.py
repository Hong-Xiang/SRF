# from .gaussFunction import *
# import numpy as np

# def test_gaussian_1d_function():
#     assert gaussian1d(0, 1, 1) == 1
#     assert gaussian1d(0, 1, 2) == 1
#     assert gaussian1d(1, 1, 1) == gaussian1d(-1, 1, 1)
    
# def test_gaussian_2d_function():
#     assert gaussian2d((0, 0), 1, 1) == 1
#     assert gaussian2d((0, 0), 1, 2) == 1
#     assert gaussian2d((1, 1), 1, 1) == gaussian2d((-1, -1), 1, 1)
    
# def test_gaussian_fit_1d():
#     N = 201
#     x = [0.01 * (i - N / 2 - 1) for i in range(N)]
#     x = np.array(x)
#     yn = [gaussian1d(t, 1, 1) for t in x]
#     yn = np.array(yn)
#     assert gaussFit1d(x, yn) == (1, 1)


# def test_gaussian_fit_2d():
#     N = 201
#     x = np.array([0.01 * (i - N / 2 - 1) for i in range(N) for j in range(N)])
#     y = np.array([0.01 * (i - N / 2 - 1) for j in range(N) for i in range(N)])
#     zn = np.array(gaussian2d((x, y), 2, 1))
#     gaussFit2d((x, y), zn)
#     assert gaussFit2d((x, y), zn) == (2, 1)

# def test_gaussian_fit_1d_solver():
#     N = 201
#     x = [0.01 * (i - N / 2 - 1) for i in range(N)]
#     x = np.array(x)
#     yn = [gaussian1d(t, 1, 1) for t in x]
#     yn = np.array(yn)
#     ans1, ans2 = gaussFit1d_solver(x, yn)
#     assert np.abs(ans1 - 1) < 0.0001 and np.abs(ans2 - 1) < 0.0001



# def test_gaussian_fit_2d_solver():
#     N = 201
#     x = np.array([0.01 * (i - N / 2 - 1) for i in range(N) for j in range(N)])
#     x = np.array(x)
#     y = np.array([0.01 * (i - N / 2 - 1) for j in range(N) for i in range(N)])
#     y = np.array(y)
#     zn = np.array(gaussian2d((x, y), 2, 1))
#     ans1, ans2 = gaussFit2d_solver((x, y), zn)
#     print(str(ans1) + '  ' + str(ans2))
#     assert np.abs(ans1 - 2) < 0.0001 and np.abs(ans2 - 1) < 0.0001


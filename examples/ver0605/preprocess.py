from srf.preprocess.preprocess import preprocess
import numpy as np

lors = np.load('lors.npy')
lors_xyz = preprocess(lors[:int(1e6), :])
np.savez('lors_tor.npz', **lors_xyz)

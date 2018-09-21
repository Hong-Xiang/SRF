import numpy as np
import time

def merge_effmap(start, end, num_rings, z_factor, path):
  """
  to do: implemented in GPU to reduce the calculated time
  """
  temp = np.load(path+'effmap_{}.npy'.format(0)).T
  num_image_layers = int(temp.shape[0])
  final_map = np.zeros(temp.shape)
  print(final_map.shape)
  st = time.time()
  # num_rings = end - start
  for ir in range(start, end):
    temp = np.load(path+'effmap_{}.npy'.format(ir)).T
    print("process :{}/{}".format(ir+1, num_rings))
    for jr in range(num_rings - ir):
      if ir == 0:
        final_map[jr:num_image_layers,:,:] += temp[0:num_image_layers-jr,:,:]
      else:
        final_map[jr:num_image_layers,:,:] += temp[0:num_image_layers-jr,:,:]
    et = time.time()
    tr = (et -st)/(num_rings*(num_rings-1)/2 - (num_rings - ir - 1)*(num_rings-ir-2)/2)*((num_rings - ir-1)*(num_rings-ir-2)/2)
    print("time used: {} seconds".format(et-st))
    print("estimated time remains: {} seconds".format(tr))

  
  # normalize the max value of the map to 1.
  cut_start = int((num_image_layers-num_rings*z_factor)/2)
  final_map = final_map[cut_start:num_image_layers-cut_start,:,:]
  final_map = final_map/np.max(final_map)
  final_map[final_map<1e-7] = 1e8
  final_map = final_map.T
  np.save(path+'summap.npy', final_map)



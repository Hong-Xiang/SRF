import numpy as np
import matplotlib.pyplot as plt
import pylab
# from scanner import Block


def preprocess_sino(sino: np.ndarray):
    npsize = sino.shape 
    temp_sino =np.zeros((npsize[1],npsize[2],npsize[0]),dtype=np.float32)
    for i in range(npsize[0]):
        for j in range(npsize[1]):
            for k in range(npsize[2]):
                temp_sino[j,k,i]=sino[i,j,k]
    # plt.figure()
    # plt.imshow(temp_sino[:,:,5])
    # pylab.show()
    sinoline = np.zeros((sino.size,1),dtype=np.float32)
    a=0
    for i in range(npsize[0]):
        for j in range(npsize[1]):
            for k in range(npsize[2]):
                #voxel = i*npsize[0]*npsize[1]+j*npsize[0]+k
                sinoline[a] = temp_sino[j,k,i] 
                a=a+1       
    return sinoline


def preprocess_X(X:np.ndarray):
    #X=x.numpy()
    npsize = X.shape
    #print(X)
    Xline = np.zeros((npsize[0]*npsize[1]*npsize[2]))
    for i in range(npsize[0]):
        for j in range(npsize[1]):
            for k in range(npsize[2]):
                voxel = k*int(npsize[0]*npsize[1])+j*int(npsize[0])+i
                Xline[voxel] = variable(X[i,j,k])            
    return Xline
    

def posprocess_X(x,shape: np.ndarray):
    X=x.numpy()
    if X.size ==shape[0,0]*shape[0,1]*shape[0,2]:
        result = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    voxel = k*shape[0]*shape[1]+j*shape[0]+i
                    result[i,j,k] = X[voxel]
        return result
    else:
        print("It's a bad trans")
    
                
    

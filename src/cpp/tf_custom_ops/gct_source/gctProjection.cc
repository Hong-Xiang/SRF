#include "gctImage.h"
// #include "hdf5/serial/hdf5.h"
#include <iostream>
using namespace std;
using namespace gct;  

ProjInfo::print()    
{
    cout << "Parameters for one angle projection data: " << endl;

    cout << "Grid size = (";
    for (int i = 0; i < ProjDim; i++)
        if (i < ProjDim - 1)
            cout << grid[i] << ", ";
        else
            cout << grid[i] << ")," << endl;

    cout << "Voxel size = (";
    for (int i = 0; i < ProjDim; i++)
        if (i < ProjDim - 1)
            cout << size[i] << ", ";
        else
            cout << size[i] << ")," << endl;

    cout << "center = (";
    for (int i = 0; i < ProjDim; i++)
        if (i < ProjDim - 1)
            cout << center[i] << ", ";
        else
            cout << center[i] << ")," << endl;
    cout << "source-to-image distance: " << sid << endl;
    cout << "source-to-axis distance: " << sad << endl;
    cout << "projection angle: " << angle << endl;
};
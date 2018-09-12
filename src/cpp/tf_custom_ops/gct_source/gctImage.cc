#include "gctImage.h"
using namespace std;
using namespace gct;

ImgInfo::print()
{
    cout << "Image parameters: " << endl;

    cout << "Grid size = (";
    for (int i = 0; i < ImgDim; i++)
        if (i < ImgDim - 1)
            cout << grid[i] << ", ";
        else
            cout << grid[i] << ")," << endl;

    cout << "Voxel size = (";
    for (int i = 0; i < ImgDim; i++)
        if (i < ImgDim - 1)
            cout << size[i] << ", ";
        else
            cout << size[i] << ")," << endl;

    cout << "center = (";
    for (int i = 0; i < ImgDim; i++)
        if (i < ImgDim - 1)
            cout << center[i] << ", ";
        else
            cout << center[i] << ")," << endl;
};
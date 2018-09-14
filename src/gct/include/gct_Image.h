/*=========================================================================
 *
 *  Copyright Guo Computed tomography Toolkit
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *=========================================================================*/


#ifndef _GCT_IMAGE_H
#define _GCT_IMAGE_H
#include <vector>
#include <iostream>
using std::cout;
using std::endl;
using std::vector;
#include <boost/multi_array.hpp>
// #include "gctImage.h"
using namespace std;


struct ImgInfo
{
    vector<int> grid;    
    vector<float> size;    
    vector<float> center;
};
// namespace gct
// {

// template <int ImgDim = 2>
// struct ImgInfo
// {
//     // explicit ImgInfo():
//     explicit ImgInfo(const vector<int>& grid, const vector<float>& size, const vector<float>& center)
//         : grid(grid), size(size), center(center) {}
    
//     explicit ImgInfo(const vector<int>& grid) 
//         : grid(grid), size(ImgDim, 1.0f), center(ImgDim, 0.0f) {}

//     explicit ImgInfo(const ImgInfo <ImgDim> &info)
//         : grid(info.grid), size(info.size), center(info.center) {}
    
//     vector<int> grid;
    
//     vector<float> size;
    
//     vector<float> center;


//     void print()
//     {
//         cout << "Image parameters: " << endl;

//         cout << "Grid size = (";
//         for (int i = 0; i < ImgDim; i++)
//             if (i < ImgDim - 1)
//                 cout << grid[i] << ", ";
//             else
//                 cout << grid[i] << ")," << endl;

//         cout << "Voxel size = (";
//         for (int i = 0; i < ImgDim; i++)
//             if (i < ImgDim - 1)
//                 cout << size[i] << ", ";
//             else
//                 cout << size[i] << ")," << endl;

//         cout << "center = (";
//         for (int i = 0; i < ImgDim; i++)
//             if (i < ImgDim - 1)
//                 cout << center[i] << ", ";
//             else
//                 cout << center[i] << ")," << endl;
//     }
// };

// template <typename float, int ImgDim = 2>
// struct Image
// {
//     typedef boost::multi_array<float, ImgDim> ImageType;

//     explicit Image(const ImgInfo<float, ImgDim> &info, const ImageType & img)
//         : info(info), values(img) {}
//     ImgInfo <float, ImgDim> info;
    
//     ImageType values;

//     void printInfo(){info.print();}

// };

// } // end namespace gct
#endif //_GCT_IMAGE_H

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

namespace gct
{

template <typename TVoxel, int ImgDim = 2>
struct ImgInfo
{
    // explicit ImgInfo();
    explicit ImgInfo(const vector<uint>& grid, const vector<TVoxel>& size, const vector<TVoxel>& center)
        : grid(grid), size(size), center(center) {}
    
    explicit ImgInfo(const vector<uint>& grid) 
        : grid(grid), size(ImgDim, 1.0f), center(ImgDim, 0.0f) {}

    explicit ImgInfo(const ImgInfo <TVoxel, ImgDim> &info)
        : grid(info.grid), size(info.size), center(info.center) {}
    
    vector<uint> grid;
    
    vector<TVoxel> size;
    
    vector<TVoxel> center;

    void print();

};

template <typename TVoxel, int ImgDim = 2>
struct Image
{
    typedef boost::multi_array<TVoxel, ImgDim> ImageType;

    explicit Image(const ImgInfo<TVoxel, ImgDim> &info, const ImageType & img)
        : info(info), values(img) {}
    ImgInfo <TVoxel, ImgDim> info;
    
    ImageType values;

    void printInfo(){info.print();}

};

} // end namespace gct
#endif //_GCT_IMAGE_H

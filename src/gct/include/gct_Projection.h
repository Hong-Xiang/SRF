/*=========================================================================
 *
 *  Copyright Guo Computed tomography Toolkit
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *=========================================================================*/


#ifndef _GCT_PROJECTION_H
#define _GCT_PROJECTION_H
#include <vector>
#include <iostream>

#include <boost/multi_array.hpp>
using namespace std;

struct ProjInfo
{
    vector <int> grid;
    vector <float> size;
    vector <float> center;
    float SID; // source to image distance
    float SAD; // source to axis distance
    float angle;
};

// namespace gct
// {

// template <int ProjDim = 1>
// struct ProjInfo // single angle data information
// {   
//     vector <int> grid;
//     vector <float> size;
//     vector <float> center;
//     float sid; // source to image distance
//     float sad; // source to axis distance
//     float angle;

//     explicit ProjInfo();
//     explicit ProjInfo(const vector<int>& grid, const vector<float>& size, 
//         const vector<float>& center, const float sid0, const float sad0, const float angle0)
//         : grid(grid), size(size), center(center), sid(sid0), sad(sad0), angle(angle0) {}
    

//     explicit ProjInfo(const ProjInfo <ProjDim> &info)
//         : grid(info.grid), size(info.size), center(info.center),
//         sid(info.sid), sad(info.sad), angle(info.angle) {}
        
//     void print()    
//     {
//         cout << "Parameters for one angle projection data: " << endl;

//         cout << "Grid size = (";
//         for (int i = 0; i < ProjDim; i++)
//             if (i < ProjDim - 1)
//                 cout << grid[i] << ", ";
//             else
//                 cout << grid[i] << ")," << endl;

//         cout << "Voxel size = (";
//         for (int i = 0; i < ProjDim; i++)
//             if (i < ProjDim - 1)
//                 cout << size[i] << ", ";
//             else
//                 cout << size[i] << ")," << endl;

//         cout << "center = (";
//         for (int i = 0; i < ProjDim; i++)
//             if (i < ProjDim - 1)
//                 cout << center[i] << ", ";
//             else
//                 cout << center[i] << ")," << endl;
//         cout << "source-to-image distance: " << sid << endl;
//         cout << "source-to-axis distance: " << sad << endl;
//         cout << "projection angle: " << angle << endl;
//     };
// };
// template <typename TPixel, int ProjDim = 1>
// struct Proj
// {
//     typedef boost::multi_array<TPixel, ProjDim> ProjType;

//     explicit Proj(const ProjInfo<ProjDim> &info, const ProjType & proj)
//         : info(info), values(proj) {}
//     ProjInfo <ProjDim> info;
    
//     ProjType values;

//     void printInfo(){info.print();}
// };

// } // end namespace gct


#endif //_GCT_PROJECTION_H
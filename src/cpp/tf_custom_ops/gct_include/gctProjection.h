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
using std::cout;
using std::endl;
using std::vector;
#include <boost/multi_array.hpp>

namespace gct
{

template <int ProjDim = 1>
struct ProjInfo // single angle data information
{   
    vector <uint> grid;
    vector <float> size;
    vector <float> center;
    float sid; // source to image distance
    float sad; // source to axis distance
    float angle;

    explicit ProjInfo(const vector<uint>& grid, const vector<float>& size, 
        const vector<float>& center, const float sid0, const float sad0, const float angle0)
        : grid(grid), size(size), center(center), sid(sid0), sad(sad0), angle(angle0) {}
    

    explicit ProjInfo(const ProjInfo <ProjDim> &info)
        : grid(info.grid), size(info.size), center(info.center),
        sid(info.sid), sad(info.sad), angle(info.angle) {}
    
    void print();
};

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


template <typename TPixel, int ProjDim = 1>
struct Proj
{
    typedef boost::multi_array<TPixel, ProjDim> ProjType;

    explicit Proj(const ProjInfo<ProjDim> &info, const ProjType & proj)
        : info(info), values(proj) {}
    ProjInfo <ProjDim> info;
    
    ProjType values;

    void printInfo(){info.print();}
};
} // end namespace gct


#endif //_GCT_PROJECTION_H
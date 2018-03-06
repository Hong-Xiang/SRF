#include "../utility/utility.hpp"

//geometries
//give the interface of base classes related to geometries

//base class of 
class vGeometry{
public:
    //output the description of the geometry.
    virtual void describe_self();
private:
    // called in initialize() to check if the geometry is valid.
    virtual bool is_valid();
};


// a point in 3D space.
class Point3D: public vGeometry{
public:
    virtual Err_i initialize(Float3 f3);
};



// A line which contains two points. 
class Line:public vGeometry{
public:
    virtual Err_i initialize(const Point3D& sp, const Point3D& ep);

    // get the start point.
    virtual Point3D& get_p_start();
    
    // get the end point.
    virtual Point3D& get_p_end();
    
    // get the length of this ray.
    virtual float get_length();
private:
    // check if the two points is overlapped
    virtual bool is_valid();
};


// unsigned int 
typedef Uint3 MeshIndex;
typedef Uint3 GridSize;

// a 3D meshgrid
class Block:public vGeometry{
public:
    virtual Err_i initialize(const GridSize& num_grid, const Float3& geometry_size, const Point3D& center);
    virtual const GridSize& get_grid();
    virtual const Float3& get_size();
    virtual const Point3D& get_center();
private:
    // check the size and grid.
    virtual bool is_valid();
};

// a surface is a convex polygon in 3D space which contains a vector of 3D points.
class Surface: public vGeometry{
public:
    // initialize a surface with a vector of points.
    virtual Err_i initialize(const std::vector<Point3D>& points);
    
    //get a certain vertex by index.
    virtual const Point3D& get_vertex(unsigned index);
    
    //get the normal of this surface.
    virtual Float3 get_normal();
    
    //check if a point is included in this surface.
    virtual bool is_in_surface(const Point3D& pt);
private:
    //check if the points are in same plane.
    virtual bool is_valid();
};

// an irregular shape with two surfaces(inner and outer) to decripe detector block. 
class Patch: public vGeometry{
public:
    virtual void initialize(const Surface& inner_surface, const Surface&  outer_surface);
    
    // get the inner or outer surface of the Patch.
    virtual const Surface& get_inner_surface();
    virtual const Surface& get_outer_surface();
    
    // get the distance between inner and outer surface.
    virtual float compute_thickness();

    // compute a box that can 
private:
    // check if the two surfaces is parallel and in different planes.
    virtual bool is_valid();
};


// a detector base class to 
class Detector: public vGeometry{
public:
    //locate a position in the discrete crystal of this block.
    virtual Err_i locate_crystal(Point3D& position);
    virtual Err_i locate_crystal(const MeshIndex& mesh, Point3D& position);
    virtual Err_i locate_crystal(unsigned index, Point3D& position);
    
    //compute the intersection length of a line(connect 2 points) with the block.
    virtual Err_i compute_intersection(const Point3D& pt1, const Point3D& pt2, float& length);
    
    //get the valid crystal point list in this detector for calculate normalization map. 
    virtual Err_i get_valid_crystals(std::vector<Point3D> point_set);
    
    //get the block of this detector. 
    virtual const Block& get_block();

};

// an monolithic detector which contain a Patch object and a block.
class MonolithicDetector: public Detector{
public:
    virtual void initialize(const Patch& patch, const Float3& pixel_size);

private:
    //get the patch shape corresponding to this detector block.
    virtual const Patch& get_patch();

    
};

// a discrete detector which is defined by block and a normal.
class DiscreteDetector: public Detector{
public:
    // directly give a 
    virtual void initialize(const Block& block, const Float3& normal);
    
};


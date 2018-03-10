/*
  @brief:  
  @auther: chengaoyu2013@gmail.com
  @date:   2018/03
*/
#include <string>
#include "../utility.h"
#include "../attributes.h"
#include "../data/data.h"

//base class of geometry
class Shape{
public: 
    virtual void describe_self();
protected:
    virtual bool is_valid() = 0;
};

//a point in 3D space.
class Point : public Shape{
public:
    virtual void initialize(float x, float y, float z);
    virtual float get_x();
    virtual float get_y();
    virtual float get_z();
};

// a line contains a start point and an end point.
class Line : public Shape{
public:
    virtual void initialize(const Point& pt0, const Point& pt1);
    virtual const Point& get_start_point();
    virtual const Point& get_end_point();
    virtual float get_length();
};

// a surface constains multiple lines in the same plane.
class Surface : public Shape{
public:
    virtual void initialize(const std::vector<Point>& points);
    virtual const Line& get_edge(unsigned index);
    virtual Float3 get_normal();
    virtual bool is_constained(const Point& point);

};

// a 3D mesh grid.
class Block : public Shape{
public:
    virtual void initialize(const Uint3& grid, const Float3& size, const Point& center);
    virtual const Uint3& get_grid();
    virtual const Float3& get_size();
    virtual const Point& get_center();
    virtual Uint3 get_total_meshes();
};

// a patch contains two parallel surfaces(inner and outer).
class Patch : public Shape{
public:
    virtual void initialize(const Surface& inner_surface, const Surface& outer_surface);
    // get the inner or outer surface of the Patch.
    virtual const Surface& get_inner_surface();
    virtual const Surface& get_outer_surface();

    virtual bool is_in_patch(const Point& pt);
    // get the distance between inner and outer surface.
    virtual float compute_thickness();
};

// a detector block that contains a block and its orientation(Float3 type).
class Detector : public Shape{
public:
    virtual const Block& get_block();
    virtual Float3 get_orientation();
    //locate a position in the discrete crystal of this block.
    virtual bool locate_crystal(Point& position);
    virtual bool locate_crystal(const Uint3& mesh, Point& position);
    virtual bool locate_crystal(unsigned index, Point& position);
    
    //compute a the insection length of a line and a detector.
    virtual float compute_intersection(const Line& line);
    
    //get the crystal positions 
    virtual std::vector<Point> get_crystal_position();
};

// an OddDetector is a detector with irregular shapes.
class OddDetector : public Detector{
public:
    //
    virtual std::vector<Point> get_crystal_position();
    virtual const Patch& get_patch();
};

// a detector Pair contains two different detectors.
class DetectorPair : public Shape{
public: 
    virtual const Detector& get_detector1();
    virtual const Detector& get_detector2();
    
    // compute the total num of lors connects the two detectors.
    virtual unsigned get_lor_num();
    // generate the lors that connect two blocks. 
    virtual Events make_lors();
};

typedef std::vector<DetectorPair> DetectorPairs;

class Scanner : public Shape{
public:
    // initialize a scanner with a file
    virtual Err_i initialize(const std::string& scanner_file);

    // get the number of the blocks in a scanner.
    virtual unsigned get_num_of_detectors();

    // locate a 3D position in a scanner and return the block index. 
    //virtual unsigned locate_detector(const Float3& position);

    // get the attributes of a scanner. Detail in ScannerAttribute.
    virtual const ScannerAttributes& get_attributes();

    // get a certain block from the block list.
    virtual const Detector& get_detector(unsigned block_index);

    // get the complete detector pair list of the whole scanner.
    virtual DetectorPairs get_detector_pair_list();
}


/*
  @brief:  
  @auther: chengaoyu2013@gmail.com
  @date:   2018/03
*/

// 
class Attributes{
public:
    virtual void print();
};

// base class to descripe a shape of a scanner
class ScannerAttributes : public Attributes{

};

//example: an CylinderPET is supposed to have:
// inner_radius, outer_radius, height, ring_gap,
// number_rings, number_blocks_per_ring.
class CylinderPETAttributes : public ScannerAttributes{
public:
    //getters and setters
private:
    float inner_radius;
    float outer_radius;
    float height;
    float ring_gap;
    unsigned number_rings;
    unsigned number_blocks_per_ring;
};
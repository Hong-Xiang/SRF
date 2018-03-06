// base class to define a sahpe of a scanner
class vScannerAttributes{

};

//example: an CylinderPET is supposed to have:
// inner_radius, outer_radius, height, ring_gap,
// number_rings, number_blocks_per_ring.
class CylinderPETAttributes:public vScannerAttributes{
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

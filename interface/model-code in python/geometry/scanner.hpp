#include <string>
#include "components.hpp"
#include "../utility/utility.hpp"
#include "scanner_attributes.hpp"
//Scanner based on blocks.
class vScanner{
public:
    // initialize a scanner with a file
    virtual Err_i initialize(const std::string& scanner_file);

    // get the number of the blocks in a scanner.
    virtual unsigned get_num_detectors();

    // locate a 3D position in a scanner and return the block index. 
    virtual unsigned locate_detector(const Float3& position);

    // get the attributes of a scanner. Detail in ScannerAttribute.
    virtual const vScannerAttributes& get_attributes();

    // get a certain block from the block list.
    virtual const Detector& get_detector(unsigned block_index);

    // print the information of this scanner
    virtual Err_i describe_self();

    // 

};

// a CylinderPET scanner derived from vScanner.
class CylinderPET : public vScanner{
public:
    // called in initialize(), create the block list
    virtual Err_i make_detector_list(const std::string& patch_file);
}
/*
  @brief:  
  @auther: chengaoyu2013@gmail.com
  @date:   2018/03
*/

// for a detector
class Detector;
class OddDetector;
class DetectorPair;

class Points; // a vecter of point.
class Lines; // a vector of line.

//get the position list of all the crystals in a detector.
Points get_crystal_positions(const Detector& detector);

Points get_crystal_positions(const OddDetector& detector);

// get all the possible lines connect the Detectorpairs.
Lines make_lors(const DetectorPair);
/*
  @brief:  
  @auther: chengaoyu2013@gmail.com
  @date:   2018/03
*/
#include<string>
// class ImageProcessor;
// class EventsProcessor;
// class Projector;
// class Scanner;
class Method;
// the base class of a task in Reconstruction.
class Task{
public:
    virtual void initialize(const std::string& config_file, const Method& method);
    // virtual Scanner* create_scanner();
    // virtual ImageProcessor* create_image_processor();
    // virtual EventsProcessor* create_events_processor();
    // virtual Projector* create_projector();
    virtual void distribute();
};

class ReconTask : public Task{
    
};

class MapTask : public Task{

}
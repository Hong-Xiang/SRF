/*
  @brief:   define the Projector used in PET reconstruction, 
            which provides basic operation such as "project" and "backproject" an event. 
  @auther: chengaoyu2013@gmail.com
  @date:   2018/03
*/

class Image;
class Event;
// base class of a projector.
class vProjector{
public:
    // project the image value along the event path.
    virtual void project(const Image& img, Event& evt);

    // backproject the event line integral to the image. 
    virtual void backproject(const Event& evt, Image& img);
}
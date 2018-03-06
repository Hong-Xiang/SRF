/*
  @brief:  this file defines the basic operators used for projection and back-projection. 
  @auther: chengaoyu2013@gmail.com
  @date:   2018/03
*/

class Image;
class Event; // can be regarded as a (x1,y1,z1,x2,y2,z2,line_integral,[weight],[tof])
class EventList; 

/// non-tof
//process one event
void project_siddon(Image* img, Event* evt);
void project_tor(Image* img, Event* evt);
void backproject_siddon(Image* img, Event* evt);
void backproject_tor(Image* img, Event* evt);


//a group of higher level kernel functions.
void project_siddon_list(Image* img, EventList* evt);
void project_tor_list(Image* img, EventList* evt);
void backproject_siddon_list(Image* img, EventList* evt);
void backproject_tor_list(Image* img, EventList* evt);

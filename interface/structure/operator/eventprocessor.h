/*
  @brief:  
  @auther: chengaoyu2013@gmail.com
  @date:   2018/03
*/
#include"../data/data.h"
class Scanner;
typedef std::vector<Events> EventsGroup;

class EventsProcessor{
public:
    //the Events may record the interaction position of Gamma with the crystal,
    //to maintain the consistence with realistic detecting process, the events are transform to the 
    //virtual discrere crystal position.
    virtual void transform_to_virtual_position(const Scanner& scanner, Events& events);

    //patition the events to several
    virtual void partition(const Events& events, EventsGroup& events_group);

    virtual void split(const Events& events, unsigned split_num, EventsGroup& events_group);
    
    virtual void merge(const EventsGroup& events_group, Events& events);
}
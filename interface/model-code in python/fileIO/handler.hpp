/*
  @brief:  
  @auther: chengaoyu2013@gmail.com
  @date:   2018/03
*/

#include<vector>
#include "./datastruct.hpp"

// a data handler is used to process the data.
class vDataHandler{
public:

};

//EventsHandler is used to seprete list-mode events data.
class EventsHandler:public vDataHandler{
public:
    // partition the events in an EventList into multiple lists in a vector.
    virtual Err_i partition(unsigned partition_num, const EventList& events, EventListVector& events_list_vector);
    
    // transform a CPU AOS EventList to GPU SOA EventArray.
    virtual Err_i toGPUArray(const EventList& events, EventArray& event_array);
}
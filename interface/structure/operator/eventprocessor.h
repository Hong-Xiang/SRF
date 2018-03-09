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
    virtual void wash(const Scanner& scanner, Events& events);

    virtual EventsGroup partition(const Events& events);
    virtual EventsGroup split(const Events& events, unsigned split_num);
    virtual Events merge(EventsGroup);
}
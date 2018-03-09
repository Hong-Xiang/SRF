/*
  @brief:  
  @auther: chengaoyu2013@gmail.com
  @date:   2018/03
*/

#include"../data/data.h"
#include"./operator.h"

class Projector : public Operator{
public:
    virtual void project(const Image& image, Events& events);
    virtual void backproject(Image& image, const Events& events);
};

class SiddonProjector : public Projector{

};

class TorProjector : public Projector{

};

/*
add more types of projector
*/
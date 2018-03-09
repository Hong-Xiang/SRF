/*
  @brief:  
  @auther: chengaoyu2013@gmail.com
  @date:   2018/03
*/

#include<string>
#include<vector>
class Line;
class Block;

class Data{

public:
    virtual unsigned get_data_size();

    virtual void load_from_file(const std::string& file_name);
    virtual void save_to_file(const std::string& file_name);
};

//a single Event
class Event{
public:
    virtual const Line& get_line();
    virtual const float& get_line_integral();
};

// an event list
class Events: public Data{
public:
    virtual Event& get_event(unsigned index);
};

class Image:public Data{
public:
    virtual const Block& get_block();
    virtual float* get_value();
};
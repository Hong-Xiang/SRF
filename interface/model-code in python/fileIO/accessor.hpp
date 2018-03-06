/*
  @brief:  file accessors are used to read and save data.
  @auther: chengaoyu2013@gmail.com
  @date:   2018/03
*/

#include "./datastruct.hpp"

// base class of file accessor
class vFileAccessor{
public:
    virtual Err_i read(const std::string& file_name, vDataStruct& data);
    virtual Err_i save(const std::string& file_name, vDataStruct& data);
};

// accessor for images
class ImageAccessor : public vFileAccessor{

};

// accessor for list-mode data
class EventListAccessor:public vFileAccessor{
  
};
/*
  @brief:  declare the data structs used for computation 
           and file accessors for reading and storage.
  @auther: chengaoyu2013@gmail.com
  @date:   2018/03
*/

#include<string>
#include<vector>
#include"../utility/utility.hpp"
class Block;

//base class of data.
class vTensor{

};


// base class of data struct.
class vDataStruct{
public:    
    virtual vTensor& get_tensor(); 
};

// 3D image which contain a block to decripe the pa
class Image:public vDataStruct{
public:
    virtual vTensor& get_tensor();
    virtual const Block& get_block();
};

// an individul event.
class Event{
public:

};

// a list of events.(list-mode data)
class EventList:public vDataStruct{

};

typedef std::vector<EventList> EventListVector;

// a struct of array data format to store events.
class EventArray:public vDataStruct{

};



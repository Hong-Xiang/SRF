/*
  @brief:  
  @auther: chengaoyu2013@gmail.com
  @date:   2018/03
*/
class Projector;

//the base class of renconsturction method model.
class Method{
public:
    virtual void intialize();
};


class IterationMethod : public Method {
public:    
    virtual 
    virtual void intialize(const Projector* projector, unsigned iteration_num);
    
    //
    virtual void iterate();

};

class Mlem : public IterationMethod{
public:
    virtual void iterate();
};

class FBP : public Method{

};


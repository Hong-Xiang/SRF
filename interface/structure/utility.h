// #include<vector>
// #include<string>


//utilities

// Error information(std::string)  
class Err_i{

};


// template class to create
template <typename T>
class Vec3{
public:
    virtual T get_vx();
    virtual T get_vy();
    virtual T get_vz();
    
    // get the value by index(0:vx, 1:vy, 2:vz, otherwise:error)
    virtual T get_value(unsigned index)
private:
    T vx;
    T vy;
    T vz;
};

// a class that contains 3 float number
typedef Vec3<float> Float3;

// a class that contains 3 unsigned int number
typedef Vec3<unsigned> Uint3;


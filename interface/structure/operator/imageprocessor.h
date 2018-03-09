/*
  @brief:  
  @auther: chengaoyu2013@gmail.com
  @date:   2018/03
*/
#include"operator.h"
class Filter;
class Image;

class ImageProcessor : public Operator{
public:
    virtual void filter(const Filter& filter, Image& image);
    virtual void add(const Image& add_image, Image& image);
    virtual void subtract(const Image& subtract_image, Image& image);
    virtual void divide(const Image& divide_image, Image& image);
    virtual void multiply(const Image& multiply_image, Image& image);
    //more images 
};


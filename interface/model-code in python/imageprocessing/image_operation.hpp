#include "../utility/utility.hpp"
class Image;
class ImageOperation{
public:
    virtual Err_i add(Image& img, const Image& op);
    virtual Err_i subtract(Image& img, const Image& op);
    virtual Err_i multiply(Image& img, const Image& op);
    virtual Err_i divide(Image& img, const Image& op);
}
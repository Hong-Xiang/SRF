
class Image;

class vFilter3D{
public:
    virtual void filter(Image& img);

};

class GaussianFilter: public vFilter3D{

};

// more kinds of filters
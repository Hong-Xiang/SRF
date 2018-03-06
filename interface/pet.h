#include "./utils.h"
#include "./brms.h"
class ImageBlock
{
    // 其他的成员变量描述具体的位置等信息
    Tensor data;
  public
    float *data_ptr();
};
class DetectorPair;
class Projection;

class TaskUnit;

class Projection;

class Image: public TaskUnit
{
  public:
    Image(ImageBlock ib, DetectorPair dp, Calculator cal){
        this.ib = ib;
        this.dp = dp;
        this.cal = cal;
    }

  private:
    ImageBlock ib;
    DetectorPair dp;

  public:
    
    void projection(Projection result)
    {
        float *res_ptr = result.data_prt();
        float *img_prt = this.ib.data_ptr();
        Postion3 bl = this.ib.bl();
        Postion3 tr = this.ib.tr();
        Postion3 d0 = this.dp.d0();
        Postion3 d1 = this.dp.d1();
        this.cal.project(res_ptr, img_ptr, bl, tr, d0, d1);
        return result;
    }

};
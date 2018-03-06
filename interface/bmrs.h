class ResultCode;

class Postion3;



class Calculator {
virtual ResultCode project(float* result, float* image, Position3 bl, Position3 tr, Position3 d0, Position3 d1);
};

class CalculatorCPU: public Calculator{
virtual ResultCode project(float* result, float* image, Position3 bl, Position3 tr, Position3 d0, Position3 d1);
};

class CalculatorGPU: public Calculator{
virtual ResultCode project(float* result, float* image, Position3 bl, Position3 tr, Position3 d0, Position3 d1);
};
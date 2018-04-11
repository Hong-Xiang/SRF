class Tensor:
    pass


class Detector:
    pass


class projection1:
    """
    所有继承这个类的类都必须是一个实现了投影的**函数**
    """
    def __call__(image, detector):
        pass


class ImageEmssion(Tensor):
    pass


class projection_seldon(projection1):
    def __call__(image, detector):
        pass




img = ImageEmssion()
detector = Detector()

# In Image.projection imple
projection_func = projection_seldon()
proj = projection_func(img, detector)

class Projection:
    """
    它是个类似C++的普通的类（不是一个函数）
    """
    def projection(self, image, detector):
        pass
    
with_projection_func_obj = Projection()
with_projection_func_obj()
proj = with_projection_func_obj.projection(image, detector)

class Interface:
    pass


class Projection(Interface):
    """
    所有继承这个接口的类都应该要有个projection方法
    """
    def projection(self):
        pass


# class ProjectionSeldon(Projection):
#     def projection(self):
#         def projection_func(image, detector):
#             pass
#         return projection_func


class ImageEmssion(Tensor, Projection):
    def projection(self, detector, model):
        return ProjectionSeldon().projection()(data, detector)


class ImageSPECT(Tensor, Projection):
    pass

class DataProjection(Tensor):
    pass
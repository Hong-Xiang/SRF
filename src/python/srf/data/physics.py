import abc


class AbstractPhysicsModel(abc.ABC):
    @abc.abstractmethod
    def projection(self, image, signal):
        pass

    @abc.abstractmethod
    def backprojection(self, signal, image):
        pass

from typing import Dict
from tensor import CompileAble, Tensor

class Host:
    def __init__(self, cluster: str, index: int):
        pass


def configurable():
    pass
class Graph(CompileAble):
    @configurable
    def __init__(self, host: Host, name: str, inputs=Dict[str, Tensor]):
        self.inputs = []
        self.ts = []
        for k in inputs:
            self.inputs[k] = inputs[k]
            self.ts['inputs/' + k] = inputs[k]
        self.host = host
        self.compile(inputs)

    def add_inputs(self, tensors):
        pass

    def add_outputs(self, tensors):
        pass

    def __call__(self, inputs=None):
        self.compile(inputs)

    def kernel(self, inputs=None):
        pass

    def compile(self, inputs=None):
        # 在这里处理tf的scope问题
        self.kernel(inputs)





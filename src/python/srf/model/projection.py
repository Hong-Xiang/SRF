from dxl.learn import Model


class Projection(Model):
    def __init__(self, name, inputs, physics):
        """
        Args:
        physics: 
        """
        self.physics = physics

    def kernel(self):
        output = self.physics.projection_op(inputs['input'])
        return output


class BackProjection(object):
    def kernel(self):
        output = self.physics.bp_op(inputs['input'])
        return output

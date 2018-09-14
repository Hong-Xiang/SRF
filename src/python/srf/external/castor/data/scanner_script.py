__all__ = ['EcatScript']

class EcatScript:
    template = 'ecat_mac.j2'
    def __init__(self, spec):
        self.spec = spec
    def render(self, template) -> str:
        return template.render(ecat=self.spec)
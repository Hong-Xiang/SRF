class Events:
    pass

class Image:
    def recursive_compile(self):
        pass

class ReconStep:
    pass

def make_compute_graph(config_file):
    events = Events()
    image = Image()
    for i in range(100):
        image = ReconStep(image, events, config_file)
    image.recursive_compile()
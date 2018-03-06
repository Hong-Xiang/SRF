class Image
    def projection(self) -> 'Projection':
        proj = Projection()
        CalculateMap.add_edge(proj, self, op.projection)

class CalculateMap:
    pass

class Projection

    def __init__(self):
        CalculateMap.add_node(self)

    def back_projction(self) -> 'Image':
        pass

img_it0 = Image()
proj_it0 = img_it0.projection()
img_it0_b = proj_it0.back_projction() / eff_map



def iter(img_init, eff_map):
    proj = img_init.projection()
    return proj.back_projction() / eff_map

def main():
    img = Image(np.zeros(img_size))
    for i in range(100):
        img=iter(img)
    session.run()
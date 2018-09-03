from jfs.api import Path
from jinja2 import Environment, PackageLoader

env = Environment(loader=PackageLoader('srf.external.castor', 'templates'))



def render(renderable):
    template = env.get_template(renderable.template)
    return renderable.render(template)

def save_cdh(path, data):
    path = Path(path)
    if path.suffix == '':
        path = path + '.Cdh'
    target = path.abs
    with open(target, 'w') as fout:
        fout.write(data)

def save_cdf():
    pass


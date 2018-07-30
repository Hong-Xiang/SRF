from jinja2 import Environment, PackageLoader

env = Environment(loader=PackageLoader('srf.external.stir', 'templates'))


def render(renderable):
    template = env.get_template(renderable.template)
    return renderable.render(template)

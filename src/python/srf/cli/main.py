import click
import json

from dxl.core.debug import enter_debug

enter_debug()

@click.group()
def srf():
    pass


@srf.command()
def recon():
    """
    Reconstruction main entry.
    """
    pass


@srf.group()
def utils():
    pass


@utils.command()
@click.argument('config', type=click.Path(exists=True))
def make_tor_lors(config):
    from ..app.srfapp import SRFApp
    click.echo('TOR Reconstruction preprocessing with config {}.'.format(config))
    with open(config, 'r') as fin:
        c = json.load(fin)
    SRFApp.make_tor_lors(c)

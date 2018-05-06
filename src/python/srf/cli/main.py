import click
import json

from ..app.srfapp import SRFApp
from dxl.core.debug import enter_debug

enter_debug()


@click.group()
def srf():
    pass


@srf.command()
@click.option('--job', '-j', type=str)
@click.option('--task-index', '-t', type=int, default=0)
@click.option('--config', '-c', type=click.Path(exists=True))
@click.option('--distribute-config', '-d', type=str)
def recon(job, task_index, config, distribute_config):
    """
    Reconstruction main entry.
    """
    with open(config, 'r') as fin:
        task_config = json.load(fin)
    if distribute_config is not None:
        with open(distribute_config, 'r') as fin:
            distribute_config = json.load(fin)
    SRFApp.reconstruction(job, task_index, task_config, distribute_config)


@srf.group()
def utils():
    pass


@utils.command()
@click.argument('config', type=click.Path(exists=True))
def make_tor_lors(config):
    """
    Preprocess to make `{x, y, z}` lors for ToR reconstruction.
    """
    click.echo('TOR Reconstruction preprocessing with config {}.'.format(config))
    with open(config, 'r') as fin:
        c = json.load(fin)
    SRFApp.make_tor_lors(c)


@utils.group()
def tor():
    pass


@tor.command()
@click.argument('source', type=click.Path(exists=True))
@click.argument('target', type=click.Path())
@click.option('--distribute-config', '-d', type=click.Path())
def auto_config(source, target, distribute_config):
    """
    Generate config for osem algorithm.
    """
    from ..app.utils.tor import auto_osem_config
    with open(source, 'r') as fin:
        c = json.load(fin)
    if distribute_config is not None:
        with open(distribute_config, 'r') as fin:
            distribute_config = json.load(fin)
    new_config = auto_osem_config(c, distribute_config)
    with open(target, 'w') as fout:
        json.dump(new_config, fout, indent=4, separators=(',', ': '))

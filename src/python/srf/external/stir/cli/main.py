import click
from srf.io import load_listmode_data 
from srf.external.stir.io import save_sinogram
from srf.external.stir.function import listmode2sinogram, position2detectorid


@click.group()
def stir():
    """
    Interface to STIR.
    """
    pass


@stir.command()
@click.option('--scanner', '-c', help='Scanner config file', type=click.types.Path(True, dir_okay=False))
@click.option('--source', '-s', help='List mode data file', type=click.types.Path(True, dir_okay=False))
@click.option('--target', '-t', help='Target file path', type=click.types.Path(False))
def lm2sino(scanner, source, target):
    data = load_listmode_data(source)
    sinogram = listmode2sinogram(scanner, data)
    save_sinogram(target, sinogram)


if __name__ == "__main__":
    stir()

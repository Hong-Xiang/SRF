import click
from ..io import load_listmode_data, save_sinogram
from ..function import listmode2sinogram



@click.group()
def stir():
    """
    Interface to STIR.
    """
    pass


@click.command()
@click.option('--scanner', '-c', help='Scanner config file', type=click.types.Path(True, dir_okay=False))
@click.option('--source', '-s', help='List mode data file', type=click.types.Path(True, dir_okay=False))
@click.option('--target', '-t', help='Target file path', type=click.types.Path(False))
def lm2sino(scanner, source, target):
    data = load_listmode_data(source)
    if isinstance(data[0], PositionEvent):
        data = data.fmap(position2detectorid)
    sinogram = listmode2sinogram(scanner, data)
    save_sinogram(target, sinogram)


if __name__ == "__main__":
    stir()

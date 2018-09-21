import click

@click.group()
def data():
    pass

@data.command()
@click.option('--source-format', '-f', help='Format of source data.')
@click.option('--target-format', '-t', help='Format of target data.')
@click.option('--target', '-o', help='Output file name.')
@click.argument('source', help='Source file name.', type=click.types.Path(exists=True))
def convert(source, source_format, target, target_format):
    source_format = source_format or guess_format(source)
    if target_format is None:
        raise TypeError("--target-format/-t is required.")
    if target is None:
        target = default_target(source, source_format, target_format)
    save(srf.data.convert(target_format, srf.data.load(source_format, source)), target)


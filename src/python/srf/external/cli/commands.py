import click


@click.group()
def external():
    pass


from ..stir.cli import stir
external.add_command(stir)

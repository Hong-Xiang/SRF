import click


@click.group()
def external():
    pass


from ..stir.cli import stir
from ..lmrec.cli import lmrec
from ..bbs.cli import bbs
from ..castor.cli import castor
external.add_command(stir)
external.add_command(lmrec)
external.add_command(bbs)
external.add_command(castor)

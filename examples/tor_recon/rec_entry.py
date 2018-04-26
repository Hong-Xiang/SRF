import click
import logging
from srf.app.srfapp import main

logger = logging.getLogger('dxl.learn.graph.reconstruction')
logger.setLevel(logging.DEBUG)



@click.command()
@click.option('--job', '-j', help='Job')
@click.option('--task', '-t', help='task', type=int, default=0)
@click.option('--config', '-c', help='config file')
def cli(job, task, config):
    main(job, task, config)

if __name__ == "__main__":
    cli()
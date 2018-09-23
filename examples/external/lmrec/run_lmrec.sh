#!/bin/bash
spack load lmrec
srf external lmrec preprocess -c /mnt/gluster/Techpi/brain16/recon/data/API1.json -t ./ -p both
srf external lmrec execute -s ./scanner_config.txt -t ./map_task.txt
srf external lmrec execute -s ./scanner_config.txt -t ./recon_task.txt
srf external lmrec postprocess -s ./ -c /mnt/gluster/Techpi/brain16/recon/data/API1.json

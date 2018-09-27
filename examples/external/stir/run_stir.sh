#!/bin/bash
srf external stir generate_data_and_header -c /mnt/gluster/Techpi/brain16/recon/data/API1.json -t ./
srf external stir generate_recon_script -c /mnt/gluster/Techpi/brain16/recon/data/API1.json -t . -s ./
srf external stir execute -c /mnt/gluster/Techpi/brain16/recon/data/API1.json -s ./
srf external stir postprocess -c /mnt/gluster/Techpi/brain16/recon/data/API1.json -s ./

#!bin/bash
spack load bbslmirp
srf external bbs preprocess -c /mnt/gluster/Techpi/brain16/recon/data/API1.json -t ./ -p both
srf external bbs execute -s ./scanner_config.txt -t ./map_task.txt
srf external bbs execute -s ./scanner_config.txt -t ./recon_task.txt
srf external bbs postprocess -s ./ -c /mnt/gluster/Techpi/brain16/recon/data/API1.json

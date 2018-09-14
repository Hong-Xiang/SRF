#!/bin/bash
srf external stir generate_data_and_header -c ../API1.json -t ./
srf external stir generate_recon_script -c ../API1.json -t . -s ./
srf external stir execute -c ../API1.json -s ./
srf external stir postprocess -c ../API1.json -s ./

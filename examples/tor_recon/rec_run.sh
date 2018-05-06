# entry_code='rec_entry.py'
python rec_entry.py -j master -c tor.json &
sleep 3
CUDA_VISIBLE_DEVICES="0" python rec_entry.py -j worker -t 0 -c tor.json &
CUDA_VISIBLE_DEVICES="1" python rec_entry.py -j worker -t 1 -c tor.json &
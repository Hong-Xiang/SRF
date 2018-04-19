import json

root  = '/home/chengaoyu/code/Python/gitRepository/SRF/debug/'
with open(root+'tor.json', 'r') as fin:
    conf = json.load(fin)
    print(type(conf['image_info']['size']))
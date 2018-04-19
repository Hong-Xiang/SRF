import json
from typing import Iterable
from srf.task.task_info import SRFTaskInfo, TorTaskInfo


root  = '/home/chengaoyu/code/Python/gitRepository/SRF/debug/'
with open(root+'tor.json', 'r') as fin:
    c = json.load(fin)
    tinfo = TorTaskInfo(c)
    print(tinfo.info['work_directory'])
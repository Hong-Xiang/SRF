from srf.task import TorTask

def create_object_class(module_name, class_name):
    class o: pass
    module_meta = __import__(module_name) 
    class_meta = getattr(module_meta, class_name) 
    o = class_meta()
    return o

if __name__ == '__main__':
    module_name = 'srf.task.tor'
    class_name = 'TorTask'
    print(create_object_class(module_name, class_name))
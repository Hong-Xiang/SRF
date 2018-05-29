class Task:
    def __init__(self, job, task_index, configure, distribution_config, scanner):
        self.scanner = make_scanner()

    def make_scanner(self):
        if self.configure['scanner_type'] == 'Ring':
            return RingPet(self.configure['scanner'])
        if self.configure['scanner_type'] == 'Patch':
            return PatchPet(self.configure['scanner'])

    def make_master(self):
        pass

    def make_worker(self):
        pass

    def run(self):
        pass


class OSEMTask(Task):
    def make_master(self):
        pass

    def make_worker(self):
        pass


class Task(Graph):
    self __init__(self, scanner, master_worker):
        self.tensors['step1'] = scanner['make_lors']

        self.tensors['step2'] = master['init']

        self.tesnors['step3'] = worker['recon']

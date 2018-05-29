def app(config):
    scanner = Scanner(config['scanner'])
    app = SRFAPP(config['xxx'], scanner)
    return app


class SRFApp:
    def __init__(self, config, job, task_index, distribution_config):
        self.configure = self.parse_config(config)
        self.scanner = self.make_scanner()
        self.task = self.make_task()

    def parse_config(self):
        pass

    def make_task(self):
        Task(config, job, task_index, dis_config)




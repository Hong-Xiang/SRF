class OSEM(Graph):
    def __init__(self, inputs: str, proj_model):
        self.projector = auto_make_projector(config)
        self.proj_model = proj_model

    def kernel(self):
        pass


class OSEMToR(Graph):
    def kernel(self):
        lors = load_tor(self.inputs['lors_files'])
        projector = TORProjector(image, self.name / 'projection')
        backprojctor = TORBackProjector(projector)


class OSEMSiddon(Graph):
    def kernel(self):
        projector = SiddonProjector(image)


class ProjectionData:
    def backpaction(self):
        pass


class WorkerGraph(Graph):
    def __init__(self, lors):
        if lors['type'] == 'TOR':
            self.lors = TORLORs(lors['file_name'])
        else:
            self.lors = LORs(lors['file_name'])

    def kernel(self):
        pass

    # construct x
    # inputs
    # recon step
    # recon step -> TORStep, SiddonStep


class WorkerGraphForLORsInXYZ(WorkerGraph):
    def __init__(self, lors, dummy):
        self.lors = TORLORs(lors)

    def kernel(self):
        class DummyTORStep(TORStep):
            pass
        if "I'm testing":
            model_class = DummyTorStep
        else:
            model_class = TORStep
        model_class(inputs['x'])
        assert model['x'] == self.inputs['x']


class WorkerGraphLORsNormal(WorkerGraph):
    def __init__(self, lors):
        self.lors = TORLORs(lors)


class WorkerGraph:
    def __init__(self):
        pass


class WorkerGraphMaker:
    def make_worker_graph(self):
        if self.config('lors')['type'] == 'TOR':
            return WorkerGraphForLORsInXYZ(self.config['lors'])
        else:
            return WorkerGraphLORsNormal(self.config['lors'])


class TORStep:
    pass

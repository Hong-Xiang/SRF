from pathlib import Path


def path_test_resources():
    import srf
    p = Path(srf.__file__)/'..'/'..'/'..'/'..'/'resource'
    return str(p.resolve())

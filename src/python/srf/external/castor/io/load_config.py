def load_config(path):
    from pathlib import Path
    import json
    with open(Path(path), 'r') as fin:
        return json.load(fin)
from pathlib import Path
import importlib
import inspect

from mykaggle.lib.registrable import Registrable


def import_features():
    current_module_path = 'mykaggle.feature'
    current_dir = Path(__file__).parent
    files = current_dir.glob('*.py')
    classes = []

    for file in files:
        print(file, f'{current_module_path}.{file.stem}')
        module = importlib.import_module(f'{current_module_path}.{file.stem}')

        # Registrable かつ file 内で定義されているものだけ追加する
        for property in dir(module):
            if not isinstance(getattr(module, property), Registrable):
                continue
            _class = getattr(module, property)
            if inspect.getfile(_class) != file:
                continue
            classes.append(_class)
    return classes


import_features()

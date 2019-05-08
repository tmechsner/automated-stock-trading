import pip
import os


def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])


if __name__ == '__main__':
    dependencies = [
        'numpy',
        'pandas',
        'sklearn',
        'scipy',
        'matplotlib',
        'enum34'
    ]

    for dep in dependencies:
        install(dep)

    string = "python SAMkNN/nearestNeighbor/setup.py install"
    os.system(string)

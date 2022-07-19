import os


def find_py_file():
    # return [elm for elm in os.listdir('.') if elm.split('.')[-1] == 'py']
    return [elm for elm in os.listdir('.') if elm.endswith('.py')]


result = find_py_file()
print(*result, sep='\n')
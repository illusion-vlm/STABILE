import datetime

def logger(file_path=None, _str=''):
    if file_path:
        _str = str(datetime.datetime.now()) + ': ' + _str
        f = open(file_path, 'a')
        f.write(_str)
        f.close()


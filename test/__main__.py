import test_all

if __name__ == '__main__':
    for name, fn in test_all.__dict__.iteritems():
        try:
            if 'test' in name: 
                fn()
        except TypeError:
            pass
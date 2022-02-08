def pretty(d, indent=0):
    """
    Pretty print a dict
    """
    for key, value in d.items():
        print('    ' * indent + str(key) + ':', end='')
        if isinstance(value, dict):
            print()
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))
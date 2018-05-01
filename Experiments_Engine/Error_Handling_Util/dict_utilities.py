
def check_dict_else_default_val(somedict, akey, default_val):
    """ Checks if a key is in a dictionary, else it adds the key to the dictionary with the default_val as entry """
    assert isinstance(somedict, dict)
    if akey in somedict.keys():
        return somedict[akey]
    else:
        somedict[akey] = default_val
        return default_val


def check_keys(somedict, somekeys):
    assert isinstance(somedict, dict)
    assert all(akey in somedict.keys() for akey in somekeys)
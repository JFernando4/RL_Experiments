
def check_dictionary_keys(keys_to_check, some_dictionary):
    if not isinstance(keys_to_check, list):
        keys_to_check = [keys_to_check]
    assert isinstance(some_dictionary, dict), "Please, provide a dictionary."
    for akey in keys_to_check:
        assert akey in some_dictionary.keys(), str(akey) + " is missing from the dictionary."

def check_parameters_types(keys_to_check, some_dictionary, type_list):
    if not isinstance(keys_to_check, list):
        keys_to_check = [keys_to_check]
    if not isinstance(type_list, list):
        type_list = [type_list]

    assert len(keys_to_check) == len(type_list), "The list must be of equal length."

    for i in range(len(keys_to_check)):
        assert isinstance(some_dictionary[keys_to_check[i]], type_list[i]), "The parameter: "\
                                                                            + str(keys_to_check[i]) +\
                                                                            ", must be of type: " + str(type_list[i]) \
                                                                            + "."

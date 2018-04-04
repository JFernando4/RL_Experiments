import os

""" Safe Checks """
def check_uniform_list_length(some_list_of_lists):
    some_length = len(some_list_of_lists[0])
    for i in range(1, len(some_list_of_lists)):
        if len(some_list_of_lists[i]) != some_length:
            return False
    return True


def check_dir_exits_and_create(somepath):
    dir_exists = os.path.exists(somepath)
    if not dir_exists:
        os.makedirs(somepath)
    return dir_exists

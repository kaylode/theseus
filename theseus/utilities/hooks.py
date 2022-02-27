import functools

def postfix_hook(function, postfunction):
    @functools.wraps(function)
    def run(*args, **kwargs):
        return postfunction(function(*args, **kwargs))
    return run
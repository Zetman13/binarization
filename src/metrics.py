import time


def timing_val(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print(f"{func.__name__} ({arg[0]}): {(t2 - t1):.4f} seconds")
        return res
    return wrapper

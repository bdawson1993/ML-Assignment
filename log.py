import time

start_time = 0


def start():
    start_time = time.perf_counter()

def stop():
    elapsedTime = time.perf_counter() - start_time

    print(f"Elapsed Time: {elapsedTime:0.4f} seconds")

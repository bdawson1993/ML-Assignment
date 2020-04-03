import time

start_time = 0
has_stopped = False


def start(text):
    if has_stopped = False:
        text += "... "
        print(text)
        start_time = time.perf_counter()
        print()
        has_stopped = True
    else:
        print("Timer hasn't been stopped...")

def stop(text):
    text += "... "
    elapsedTime = time.perf_counter() - start_time
    print(text + f"Elapsed Time: {elapsedTime:0.4f} seconds")
    print()
    has_stopped = True

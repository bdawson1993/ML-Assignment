import time
start_time = 0

def start(text):
        text += "... "
        print(text)
        start_time = 0
        print()
        
   

def stop(text):
    text += "... "
    elapsedTime = time.perf_counter() - start_time
    print(text + f"Elapsed Time: {elapsedTime:0.4f} seconds")
    print()
    


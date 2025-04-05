import threading
import os
import concurrent.futures
from multiprocessing import Process

 
def task1():
    print("Task 1 assigned to thread: {}".format(threading.current_thread().name))
    print("ID of process running task 1: {}".format(os.getpid()))
 
def task2():
    print("Task 2 assigned to thread: {}".format(threading.current_thread().name))
    print("ID of process running task 2: {}".format(os.getpid()))
 
def worker(name):
    print(f"Worker thread running: {name}")
 
def print_func(continent='Asia'):
    print('The name of continent is : ', continent)



if __name__ == "__main__":
    print("ID of process running main program: {}".format(os.getpid()))
 
    print("Main thread name: {}".format(threading.current_thread().name))
 
    # Create two separate threads
    t1 = threading.Thread(target=task1, name='t1')
    t2 = threading.Thread(target=task2, name='t2')
 
    # Start the threads
    t1.start()
    t2.start()
 
    # End the thread executions
    t1.join()
    t2.join()


    #########################################################
    mw=2 #max workers
    print()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=mw)

    pid=os.getpid()
    for i in range(mw):
        pool.submit(worker(pid))
    
    pool.shutdown(wait=True)
    
    print("Main thread continuing to run")

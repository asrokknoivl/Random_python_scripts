
import threading
import time
import schedule
from datetime import datetime

print(threading.current_thread().name)
print(threading.main_thread().name)

def task1(name):
    print('ddddddddddd')
    print('next dddddddddd in 30')
    time.sleep(30)
    

def task2():
    while True:
        print('2')

# th1 = threading.Thread(target= task1,args=('hey',))
# th2 = threading.Thread(target= task2)
# th1.start()
# th2.start()
# th1.join()
# while True:
#     if datetime.today().minute in range(30, 40):
#         schedule.every(2).minutes.do(task2)
#     if datetime.today().minute in range(40, 50):
#         schedule.every(4).minutes.do(task2)
#     if datetime.today().minute in range(50, 60):
#         schedule.every(3).minutes.do(task2)
#     if datetime.today().minute in range(0, 10):
#         schedule.every(8).minutes.do(task2)
#     schedule.run_pending()
#

schedule.every(10).seconds.do(task2)
schedule.clear()
schedule.every(20).seconds.do(task2)
print(schedule.jobs)

















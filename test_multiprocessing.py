import multiprocessing
import numpy as np
import time

def compute_box(i,Q):
#    print("compute_box",time.localtime( time.time() ))
#    time.sleep(0.3)
    Q.put_nowait(i)
t1 = time.time()
num_boxes=23
Q = multiprocessing.Queue(num_boxes)
p_list = []
tc = time.time()
for i in range(num_boxes):
#    print("create start",time.localtime( time.time() ))
    p = multiprocessing.Process(target=compute_box,args=(i,Q))
    p.start()
    p_list.append(p)
td = time.time()
print("crete process tiem is ",td - tc)
b=0
img_boxes = np.zeros(num_boxes)
#print("Q.size",Q.qsize())
#time.sleep(0.32)
for p in p_list:
#    print("b = ",b)
    ta = time.time()
    img_boxes[b] = Q.get()
    tb = time.time()
#    print("get data time ", tb - ta)
#    print("img_boxes[b] ",img_boxes[b])
    b += 1
#    p.join()
t2 = time.time()
print("get data total time ",t2 - td)
print("time is ",t2 - t1)

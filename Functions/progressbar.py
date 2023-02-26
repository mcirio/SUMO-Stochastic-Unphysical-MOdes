import sys
import time

def progressbar(it, prefix="", size=60, file=sys.stdout):
    t_1 = time.time()
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    t_2 = time.time()
    print("Time (min): ",(t_2-t_1)/60.)
    #file.write("\n")
    file.flush()
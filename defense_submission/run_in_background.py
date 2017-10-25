from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Queue
import threading

def generate_in_background(generator, num_cached=50):
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()

    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()

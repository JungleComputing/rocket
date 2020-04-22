package nl.esciencecenter.rocket.util;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayDeque;
import java.util.concurrent.BlockingDeque;

public class FutureQueue<T> {
    protected static final Logger logger = LogManager.getLogger();

    private ArrayDeque<T> items;
    private ArrayDeque<Future<T>> waiters;

    public FutureQueue() {
        items = new ArrayDeque<>();
        waiters = new ArrayDeque<>();
    }

    public synchronized T popBlocking() {
        return items.poll();
    }

    public synchronized Future<T> popAsync() {
        Future<T> fut;
        T item;

        if ((item = items.poll()) != null) {
            fut = Future.ready(item);
        } else {
            fut = new Future<>();
            waiters.push(fut);
        }

        return fut;
    }

    public synchronized void push(T item) {
        Future<T> fut;

        if ((fut = waiters.poll()) != null) {
            fut.complete(item);
        } else {
            items.add(item);
        }
    }
}

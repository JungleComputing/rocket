package nl.esciencecenter.rocket.util;

import java.util.concurrent.atomic.AtomicLong;

public class PriorityRunnable implements Runnable, Comparable<PriorityRunnable> {
    static final AtomicLong seq = new AtomicLong(0);

    static final public long PRIORITY_HIGHEST = 400;
    static final public long PRIORITY_HIGH = 300;
    static final public long PRIORITY_MID = 200;
    static final public long PRIORITY_LOW = 100;
    static final public long PRIORITY_LOWEST = 0;

    private Runnable obj;
    private long time;
    private long priority;

    public PriorityRunnable(long priority, Runnable obj) {
        this.obj = obj;
        this.time = seq.getAndIncrement();
        this.priority = priority;
    }

    @Override
    public void run() {
        obj.run();
    }

    @Override
    public int compareTo(PriorityRunnable that) {
        if (this.priority != that.priority) {
            return -Long.compare(this.priority, that.priority);
        } else {
            return Long.compare(this.time, that.time);
        }
    }
}

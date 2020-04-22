package nl.esciencecenter.rocket.cache;

import nl.esciencecenter.rocket.util.Future;
import nl.esciencecenter.rocket.util.LRUQueue;
import nl.esciencecenter.rocket.util.Util;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.lang.reflect.Array;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

public abstract class AbstractCache<B, M> {
    protected static final Logger logger = LogManager.getLogger();

    // State of cache entry
    enum EntryStatus {
        // Only one thread has access which is writing to the slot.
        WRITING,

        // Multiple threads have access to the slot which can all read its content.
        READONLY,

        // Indicates that the slot is still alive but its content is not valid. This happens when one thread had
        // writable access but released the entry without calling downgradeToReader.
        INVALID,

        // Cache slot has been deleted. Accessing a cache entry with this status is an error.
        DELETED,
    }

    public class Entry {
        // Read-only fields, these are never changed (although buffer itself can be mutated)
        private String key;
        private B buffer;

        // These fields require holding the entry's monitor to access
        private M meta;
        private int refCount;
        private EntryStatus status;
        private Deque<Transaction> pendingReaders;

        public Entry(String key, B buffer) {
            this.key = key;
            this.buffer = buffer;
            this.refCount = 0;
            this.status = EntryStatus.INVALID;
            this.pendingReaders = new ArrayDeque<>();

            lru.addFirst(key);
        }

        public String getKey() {
            return key;
        }

        public boolean isWriter() {
            synchronized (this) {
                return status == EntryStatus.WRITING;
            }
        }

        public boolean isReader() {
            synchronized (this) {
                return status == EntryStatus.READONLY;
            }
        }

        public M size() {
            synchronized (this) {
                if (!isReader()) {
                    throw new IllegalStateException("cannot obtain size data while entry is writable");
                }

                return meta;
            }
        }

        public B read() {
            return buffer;
        }


        public B write() {
            synchronized (this) {
                if (status != EntryStatus.WRITING) {
                    throw new IllegalStateException("cannot write to read-only entry");
                }
            }

            return buffer;
        }

        public void downgradeToReader(M size) {
            Deque<Transaction> oldPendingReaders;

            synchronized (this) {
                if (status != EntryStatus.WRITING) {
                    throw new IllegalArgumentException("cannot downgrade cache entry, entry is not writable");
                }

                this.meta = size;
                this.status = EntryStatus.READONLY;

                oldPendingReaders = this.pendingReaders;
                this.pendingReaders = new ArrayDeque<>();
            }

            while (!oldPendingReaders.isEmpty()) {
                oldPendingReaders.pop().trigger();
            }
        }

        protected boolean acquireImmediately() {
            // Ordering of locks is important. First the parent lock then the entry's lock to prevent deadlocking.
            synchronized (AbstractCache.this) {
                synchronized (this) {
                    if (status != EntryStatus.READONLY) {
                        return false;
                    }

                    if (refCount++ == 0) {
                        lru.remove(key);
                    }

                    return true;
                }
            }
        }

        protected void acquire(Transaction waiter) {

            // Ordering of locks is important. First the parent lock then the entry's lock to prevent deadlocking.
            synchronized (AbstractCache.this) {
                synchronized (this) {
                    switch (status) {
                        // Exclusive, some other thread is writing to this entry. Put ourselves in the pending
                        // list and wait until the writer is done.
                        case WRITING:
                            pendingReaders.add(waiter);
                            waiter = null; // set to null since we don't want to trigger it now.
                            break;

                        // Shared, multiple threads are reading from this entry. We immediately complete the given
                        // future since the entry is available.
                        case READONLY:
                            // nothing to do
                            break;

                        // Abondoned, some other thread had exclusive access but released it before finishing writing.
                        // We become the exclusive owner of this entry now.
                        case INVALID:
                            status = EntryStatus.WRITING;
                            break;

                        // Impossible
                        default:
                            throw new IllegalStateException("internal error, cache entry has invalid state");
                    }

                    // Increase the refcount. If it was previously 0, we are the first so remove it from the LRU
                    refCount++;
                    if (refCount == 1) {
                        lru.remove(key);
                    }
                }
            }

            // Trigger waiter outside of synchronized statement
            if (waiter != null) {
                waiter.trigger();
            }
        }

        public void release() {
            Transaction waiter = null;

            // Ordering of locks is important. First the parent lock then the entry's lock to prevent deadlocking.
            synchronized (AbstractCache.this) {
                synchronized (this) {
                    // Cannot release if refcount is 0. Invalid state?
                    if (refCount <= 0) {
                        throw new IllegalArgumentException("cannot release cache entry since refcount is zero");
                    }

                    switch (status) {
                        // Exclusive, we released this entry without fulfilling it. There are two options:
                        //   1. If there are readers waiting, release the first.
                        //   2. Otherwise, set the state to invalid
                        case WRITING:
                            waiter = pendingReaders.pollFirst();
                            if (waiter == null) {
                                status = EntryStatus.INVALID;
                            }
                            break;

                        // Shared. No action required.
                        case READONLY:
                            // Nothing to do
                            break;

                        // Impossible
                        case INVALID:
                        case DELETED:
                            throw new IllegalStateException("internal error, cache entry has invalid state");
                    }

                    // Decrease the refcount. If it is now 0, we are the last so add it to the LRU
                    refCount--;
                    if (refCount == 0) {
                        if (status == EntryStatus.READONLY) {
                            lru.addLast(key);
                        } else {
                            lru.addFirst(key);
                        }

                        makeProgress();
                    }
                }
            }

            // Trigger waiter outside of synchronized statement
            if (waiter != null) {
                waiter.trigger();
            }
        }

        @Override
        public String toString() {
            return "Slot{" +
                    "key='" + key + '\'' +
                    ", buffer=" + (buffer != null ? "<...>" : "null") +
                    ", refCount=" + refCount +
                    ", status=" + status +
                    '}';
        }
    }

    class EntrySet {

    }

    class Transaction {
        private String[] keys;
        private Entry[] entries;
        private Future<Entry[]> future;
        private AtomicInteger pending;

        @SuppressWarnings("unchecked")
        public Transaction(String[] keys) {
            this.keys = removeDuplicates(keys);
            this.entries = (Entry[]) Array.newInstance(Entry.class, this.keys.length);
            this.pending = new AtomicInteger(this.keys.length);
            this.future = new Future<>();
        }

        public void trigger() {
            if (pending.decrementAndGet() == 0) {
                future.complete(entries);
            }
        }
    }

    private ArrayDeque<Transaction> pendingTransactions;
    private HashMap<String, Entry> cache;
    private LRUQueue<String> lru;

    protected AbstractCache() {
        pendingTransactions = new ArrayDeque<>();
        cache = new HashMap<>();
        lru = new LRUQueue<>();
    }

    protected abstract Optional<B> createBuffer(String key);
    protected abstract void destroyBuffer(B buffer);

    synchronized public Optional<Entry> tryAcquireImmediately(String key) {
        Entry e = cache.getOrDefault(key, null);
        if (e != null && e.acquireImmediately()) {
            return Optional.of(e);
        } else {
            return Optional.empty();
        }
    }

    synchronized public Future<Entry[]> acquireAllAsync(String ... keys) {
        Transaction trans = new Transaction(keys);
        pendingTransactions.addLast(trans);
        makeProgress();

        return trans.future;
    }

    synchronized public Future<Entry> acquireAsync(String key) {
        String[] keys = new String[]{key};
        Future<Entry[]> fut = acquireAllAsync(keys);

        return fut.thenMap(m -> m[0]);
    }

    synchronized private void makeProgress() {
        boolean done = false;

        while (!done && !pendingTransactions.isEmpty()) {
            Transaction trans = pendingTransactions.removeFirst();
            boolean success = true;

            for (int i = 0; i < trans.keys.length; i++) {
                String key = trans.keys[i];
                Entry e = trans.entries[i];

                if (e != null) {
                    continue;
                }

                e = cache.getOrDefault(key, null);

                if (e != null) {
                    trans.entries[i] = e;
                    e.acquire(trans);
                    continue;
                }

                Optional<B> buffer = createBuffer(key);

                if (buffer.isPresent()) {
                    e = new Entry(key, buffer.get());
                    cache.put(key, e);
                    trans.entries[i] = e;

                    e.acquire(trans);
                    continue;
                }

                success = false;
            }

            if (!success) {
                pendingTransactions.addFirst(trans);
                done = !evictOne();
            }
        }
    }

    synchronized private boolean evictOne() {
        // delete unused cache item and free the buffer so it can be reused
        Optional<String> oldest = lru.removeFirst();
        if (oldest.isEmpty()) {
            return false;
        }

        B buffer;
        Entry entry = cache.remove(oldest.get());

        synchronized (entry) {
            // Sanity check
            if (entry.refCount != 0 ||
                    entry.status == EntryStatus.DELETED ||
                    entry.pendingReaders.size() > 0) {
                throw new IllegalStateException("internal error, cannot delete invalid cache entry: " +
                        entry);
            }

            // delete entry
            buffer = entry.buffer;
            entry.buffer = null;
            entry.status = EntryStatus.DELETED;
        }

        destroyBuffer(buffer);

        return true;
    }

    synchronized public void cleanup() {
        while (evictOne()) {
            //
        }

        if (!lru.isEmpty() || !cache.isEmpty()) {
            throw new IllegalStateException("failed to clean up cache, " + cache.size() + " entries are still in use!");
        }
    }

    protected static String[] removeDuplicates(String[] keys) {
        String[] output = new String[keys.length];
        int n = 0;

        for (String key: keys) {
            boolean found = false;

            for (int i = 0; i < n; i++) {
                found |= output[i].equals(key);
            }

            if (!found && key != null) {
                output[n++] = key;
            }
        }

        return output.length != n ? Arrays.copyOf(output, n) : output;
    }
}

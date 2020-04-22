package nl.esciencecenter.rocket.util;

import java.lang.ref.WeakReference;
import java.util.WeakHashMap;

public class InternPool<T> {
    private final WeakHashMap<T, WeakReference<T>> pool = new WeakHashMap<T, WeakReference<T>>();

    public synchronized T intern(T object) {
        WeakReference<T> ref = pool.get(object);
        if (ref == null) {
            pool.put(object, new WeakReference<T>(object));
            return object;
        } else {
            return ref.get();
        }
    }
}
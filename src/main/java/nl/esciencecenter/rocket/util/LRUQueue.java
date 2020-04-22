package nl.esciencecenter.rocket.util;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
import java.util.Optional;

public class LRUQueue<K> {
    protected static final Logger logger = LogManager.getLogger();

    private class Item {
        private K key;
        private Item next;
        private Item prev;
    }

    private HashMap<K, Item> items;
    private Item firstItem;
    private Item lastItem;

    public LRUQueue() {
        items = new HashMap<>();
        firstItem = null;
        lastItem = null;
    }

    public void addLast(K key) {
        if (items.containsKey(key)) {
            throw new IllegalArgumentException("queue cannot contain duplicate entries");
        }

        Item item = new Item();
        item.key = key;
        items.put(key, item);

        if (lastItem == null) {
            lastItem = item;
            firstItem = item;
        } else {
            item.prev = lastItem;
            lastItem.next = item;
            lastItem = item;
        }
    }

    public void addFirst(K key) {
        if (items.containsKey(key)) {
            throw new IllegalArgumentException("queue cannot contain duplicate entries");
        }

        Item item = new Item();
        item.key = key;
        items.put(key, item);

        if (firstItem == null) {
            lastItem = item;
            firstItem = item;
        } else {
            item.next = firstItem;
            firstItem.prev = item;
            firstItem = item;
        }
    }

    private void removeFromLinkedList(Item item) {
        if (item.prev == null) {
            firstItem = item.next;
        } else {
            item.prev.next = item.next;
        }

        if (item.next == null) {
            lastItem = item.prev;
        } else {
            item.next.prev = item.prev;
        }

        item.next = null;
        item.prev = null;
    }

    public Optional<K> removeFirst() {
        if (firstItem != null) {
            K key = firstItem.key;
            remove(key);
            return Optional.of(key);
        } else {
            return Optional.empty();
        }
    }

    public void remove(K key) {
        Item item = items.remove(key);
        if (item == null) {
            throw new IllegalArgumentException("key not found");
        }

        removeFromLinkedList(item);
    }

    public boolean isEmpty() {
        return items.isEmpty();
    }

    public int size() {
        return items.size();
    }
}

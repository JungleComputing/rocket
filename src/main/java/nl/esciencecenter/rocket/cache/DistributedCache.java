package nl.esciencecenter.rocket.cache;

import ibis.ipl.ConnectionClosedException;
import ibis.ipl.IbisIdentifier;
import ibis.ipl.ReadMessage;
import ibis.ipl.WriteMessage;
import nl.esciencecenter.rocket.profiling.Profiler;
import nl.esciencecenter.rocket.types.HashableKey;
import nl.esciencecenter.rocket.util.Future;
import nl.junglecomputing.pidgin.ExplicitChannel;
import nl.junglecomputing.pidgin.Pidgin;
import nl.junglecomputing.pidgin.PidginFactory;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Properties;
import java.util.Queue;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

public class DistributedCache {
    protected static final Logger logger = LogManager.getLogger();

    class SenderThread extends Thread {
        private IbisIdentifier remoteRank;
        private Queue<Object> inbox;
        private Queue<Object> outbox;
        private boolean shutdown;

        private SenderThread(IbisIdentifier peer) {
            setName("distributed-cache-sender-" + peer);
            this.remoteRank = peer;
            this.inbox = new ArrayDeque<>();
            this.outbox = new ArrayDeque<>();
            this.shutdown = false;
        }

        private void pushIncomingMessage(Object msg) {
            synchronized (this) {
                inbox.add(msg);
                notify();
            }
        }

        private void pushOutgoingMessage(Object msg) {
            synchronized (this) {
                outbox.add(msg);
                notify();
            }
        }

        public void shutdown() {
            synchronized (this) {
                shutdown = true;
                notify();
            }

            try {
                join();
            } catch (InterruptedException e) {
                // ?
            }
        }

        @Override
        public void run() {
            try {
                while (true) {
                    Object inMsg = null;
                    Object outMsg = null;

                    synchronized (this) {
                        if (!outbox.isEmpty()) {
                            outMsg = outbox.poll();
                        } else if (!inbox.isEmpty()) {
                            inMsg = inbox.poll();
                        } else if (shutdown) {
                            break;
                        } else {
                            try {
                                wait();
                            } catch (InterruptedException e) {
                                //
                            }
                        }
                    }

                    // Send out message
                    if (outMsg != null) {
                        sendMessage(outMsg);
                    }

                    // Processing in message
                    if (inMsg != null) {
                        processingIncomingMessage(inMsg);
                    }
                }
            } catch (Throwable e) {
                logger.error("sender thread failed", e);
            }
        }

        private void processingIncomingMessage(Object obj) {
            // Handle incoming request
            if (obj instanceof RequestMsg) {
                RequestMsg msg = (RequestMsg) obj;

                List<IbisIdentifier> candidates = getAndPushCandidate(msg.key, remoteRank);
                ForwardMsg fwd = new ForwardMsg(msg.id, msg.key, remoteRank, candidates, 0);

                forwardRequest(fwd);
                return;
            }

            // Handle incoming forward
            if (obj instanceof ForwardMsg) {
                ForwardMsg msg = (ForwardMsg) obj;
                HostCache.Entry entry = hostCache.tryAcquireImmediately(msg.key).orElse(null);

                if (entry != null) {
                    profiler.report("distributed-cache-sender", "distributed cache hit after " +
                            msg.hopCount + " hops");

                    try {
                        ByteBuffer buffer = entry.read().asByteBuffer().duplicate();
                        buffer.position(0);
                        buffer.limit(entry.size().intValue());

                        sendMessageWithBuffer(new ResponseMsg(msg.id, true), buffer);
                    } finally {
                        entry.release();
                    }
                } else {
                    forwardRequest(msg);
                }

                return;
            }

            // Handle incoming response
            if (obj instanceof ResponseMsg) {
                ResponseMsg msg = (ResponseMsg) obj;
                LocalPendingRequest req = requests.remove(msg.id);

                if (req == null) {
                    logger.warn("received unexpected response, ignoring message (id: {})", msg.id);
                    return;
                }

                req.future.complete(Optional.empty());
                return;
            }

            logger.error("could not handle message of type {}", obj.getClass().getSimpleName());
        }

        private void forwardRequest(ForwardMsg msg) {
            List<IbisIdentifier> nextHop = msg.nextHop;

            if (nextHop.size() > 0) {
                IbisIdentifier target = nextHop.remove(0);
                ForwardMsg fwd = new ForwardMsg(
                        msg.id,
                        msg.key,
                        msg.originalSource,
                        nextHop,
                        msg.hopCount + 1);

                logger.trace("forwarding request for {} to next node {} (id: {})", msg.key, target, msg.id);
                senders.get(target).pushOutgoingMessage(fwd);
            } else {
                logger.trace("sending failure response for {} to {} (id: {})", msg.key, remoteRank, msg.id);
                sendMessage(new ResponseMsg(msg.id, false));
            }
        }

        private void sendMessage(Object msg) {
            if (remoteRank.equals(myRank)) {
                if (msg instanceof RequestMsg || msg instanceof ResponseMsg) {
                    processingIncomingMessage(msg);
                } else if (msg instanceof ForwardMsg) {
                    senders.get(((ForwardMsg) msg).originalSource).pushIncomingMessage(msg);
                } else {
                    logger.error("could not handle message of type {}", msg.getClass().getSimpleName());
                }

                return;
            }

            WriteMessage m = null;
            try {
                m = channel.sendMessage(remoteRank);
                m.writeObject(msg);
                m.finish();
            } catch (IOException e) {
                if (m != null) m.finish(e);
                logger.warn("failed to send message to {}:", remoteRank, e);
            }
        }

        private void sendMessageWithBuffer(ResponseMsg msg, ByteBuffer buffer) {
            if (remoteRank.equals(myRank)) {
                sendMessage(new ResponseMsg(msg.id, false));
                return;
            }

            WriteMessage m = null;
            try {
                m = channel.sendMessage(remoteRank);
                m.writeObject(msg);
                m.writeInt(buffer.remaining());
                m.writeByteBuffer(buffer);
                m.finish();
            } catch (IOException e) {
                if (m != null) m.finish(e);
                logger.warn("failed to send response to {}:", remoteRank, e);
            }
        }
    }


    class ReceiverThread extends Thread {
        private IbisIdentifier remoteRank;
        private boolean shutdown;

        private ReceiverThread(IbisIdentifier peer) {
            setName("distributed-cache-receiver-" + peer);
            remoteRank = peer;
            shutdown = false;
        }

        private void shutdown() {
            synchronized (this) {
                shutdown = true;
            }
        }

        public void run() {
            ReadMessage m = null;
            try {
                while (true) {
                    m = channel.receiveMessage(remoteRank);
                    if (m == null) {
                        break;
                    }

                    Object obj = m.readObject();

                    if (obj instanceof RequestMsg) {
                        handleRequest((RequestMsg) obj);
                    } else if (obj instanceof ForwardMsg) {
                        handleForward((ForwardMsg) obj);
                    } else if (obj instanceof ResponseMsg) {
                        handleResponse((ResponseMsg) obj, m);
                    } else {
                        throw new RuntimeException("received unexpected message: " + obj.getClass().getSimpleName());
                    }

                    m.finish();
                    m = null;
                }
            } catch (Exception e) {
                if (m != null && e instanceof IOException)
                    m.finish((IOException) e);

                if  (!shutdown || !(e instanceof ConnectionClosedException)) {
                    logger.error("error when receiving message", e);
                }
            }
        }

        private void handleRequest(RequestMsg msg) {
            logger.trace("received request for {} from {} (id: {}", msg.key, remoteRank, msg.id);
            senders.get(remoteRank).pushIncomingMessage(msg);
        }

        private void handleForward(ForwardMsg msg) {
            logger.trace("received forward for {} from {} (id: {}, hops: {} next hop: {})",
                    msg.key, remoteRank, msg.id, msg.hopCount, msg.nextHop);
            senders.get(msg.originalSource).pushIncomingMessage(msg);
        }

        private void handleResponse(ResponseMsg msg, ReadMessage m) throws IOException {
            LocalPendingRequest req = requests.remove(msg.id);
            if (req == null) {
                logger.warn("received unexpected response, ignoring message (id: {})", msg.id);
                return;
            }

            logger.trace("received response for {} from {} (id: {}, hit: {})",
                    req.key, remoteRank, msg.id, msg.hit);

            if (msg.hit) {
                int size = m.readInt();
                ByteBuffer buf = req.buffer;
                buf.position(0);
                buf.limit(size);

                m.readByteBuffer(buf);
                req.future.complete(Optional.of((long) size));
            } else {
                req.future.complete(Optional.empty());
            }
        }
    }

    static private class LocalPendingRequest {
        final private UUID id;
        final private HashableKey key;
        final private ByteBuffer buffer;
        final private Future<Optional<Long>> future;

        private LocalPendingRequest(HashableKey key, ByteBuffer buffer) {
            this.key = key;
            this.buffer = buffer;
            this.id = UUID.randomUUID();
            this.future = new Future<>();
        }
    }

    static private class RequestMsg implements Serializable  {
        final private UUID id;
        final private HashableKey key;

        private RequestMsg(UUID id, HashableKey key) {
            this.id = id;
            this.key = key;
        }
    }

    static private class ResponseMsg implements Serializable {
        final private UUID id;
        final private boolean hit;

        private ResponseMsg(UUID id, boolean hit) {
            this.id = id;
            this.hit = hit;
        }
    }

    static private class ForwardMsg implements Serializable {
        final private UUID id;
        final private HashableKey key;
        final private IbisIdentifier originalSource;
        final private List<IbisIdentifier> nextHop;
        final private int hopCount;

        private ForwardMsg(UUID id, HashableKey key, IbisIdentifier originalSource, List<IbisIdentifier> nextHop, int hopCount) {
            this.id = id;
            this.key = key;
            this.originalSource = originalSource;
            this.nextHop = nextHop;
            this.hopCount = hopCount;
        }
    }

    private final int maxHops;
    private final IbisIdentifier[] peers;
    private final IbisIdentifier myRank;
    private final String pidginName;
    private final ExplicitChannel channel;
    private final Map<IbisIdentifier, SenderThread> senders;
    private final Map<IbisIdentifier, ReceiverThread> receivers;
    private final ConcurrentMap<UUID, LocalPendingRequest> requests;

    private Profiler profiler;
    private HostCache hostCache;

    private final HashMap<HashableKey, List<IbisIdentifier>> candidates;

    public DistributedCache(String hostName, int mxHops) throws Exception {
        pidginName = "distcache_" + System.getProperty("ibis.pool.name");

        Properties props = new Properties();
        props.put("ibis.server.address", System.getProperty("ibis.server.address"));
        props.put("ibis.server.port", System.getProperty("ibis.server.port"));
        props.put("ibis.pool.size", System.getProperty("ibis.pool.size"));
        props.put("ibis.pool.name", pidginName);
        props.put("ibis.implementation", "ib");

        Pidgin pidgin = PidginFactory.create(pidginName, props);
        peers = pidgin.getAllIdentifiers();
        myRank = pidgin.getMyIdentifier();
        candidates = new HashMap<>();
        maxHops = mxHops;

        requests = new ConcurrentHashMap<>();
        receivers = new HashMap<>();
        senders = new HashMap<>();
        channel = pidgin.createExplicitChannel(pidginName);

        for (IbisIdentifier peer : peers) {
            senders.put(peer, new SenderThread(peer));
        }

        for (IbisIdentifier peer : peers) {
            if (!peer.equals(myRank)) {
                receivers.put(peer, new ReceiverThread(peer));
            }
        }
    }

    private IbisIdentifier getOwner(HashableKey key) {
        return peers[Math.abs(key.hashCode()) % peers.length];
    }

    private List<IbisIdentifier> getAndPushCandidate(HashableKey key, IbisIdentifier source) {
        synchronized (candidates) {
            List<IbisIdentifier> oldList = candidates.getOrDefault(key, new ArrayList<>());
            List<IbisIdentifier> newList = new ArrayList<>(maxHops);

            newList.add(source);
            for (IbisIdentifier node: oldList) {
                if (newList.size() < maxHops && !newList.contains(node)) {
                    newList.add(node);
                }
            }

            candidates.put(key, newList);
            return oldList;
        }
    }

    public synchronized void activate(HostCache hostCache, Profiler prof) throws IOException {
        this.hostCache = hostCache;
        this.profiler = prof;

        channel.activate();

        for (SenderThread sender: senders.values()) {
            sender.start();
        }

        for (ReceiverThread receiver: receivers.values()) {
            receiver.start();
        }
    }


    public Future<Optional<Long>> request(HashableKey key, ByteBuffer destBuffer) {
        LocalPendingRequest req = new LocalPendingRequest(key, destBuffer);
        requests.put(req.id, req);

        IbisIdentifier target = getOwner(key);
        senders.get(target).pushOutgoingMessage(new RequestMsg(req.id, key));

        return req.future;
    }


    public synchronized void shutdown() {
        for (SenderThread sender: senders.values()) {
            sender.shutdown();
        }

        for (ReceiverThread receiver: receivers.values()) {
            receiver.shutdown();
        }

        try {
            channel.deactivate();
        } catch (IOException e) {
            logger.warn("failed to deactive channel", e);
        }

        try {
            PidginFactory.terminate(pidginName);
        } catch (IOException e) {
            logger.warn("failed to terminate Pidgin", e);
        }
    }

}
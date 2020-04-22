package nl.esciencecenter.rocket.cache;

import jcuda.CudaException;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayDeque;
import java.util.Optional;

public class DeviceCache extends AbstractCache<CudaMemByte, Long> {
    protected static final Logger logger = LogManager.getLogger();
    private ArrayDeque<CudaMemByte> buffers;
    private long numEntries;
    private long entrySize;


    public DeviceCache(CudaContext ctx, long totalSize, long bufferSize) throws CudaException {
        buffers = new ArrayDeque<>();
        numEntries = 0;
        entrySize = bufferSize;
        long initialFree = ctx.getFreeMemory();
        long currentlyFree = initialFree;

        try {
           while (initialFree - currentlyFree + bufferSize < totalSize) {
                buffers.push(ctx.allocBytes((int) bufferSize));
                numEntries++;
                currentlyFree = ctx.getFreeMemory();
            }
        } catch (CudaException e) {
            logger.warn("attempted to allocate {} bytes on device {}, but could only allocate {} bytes: {}",
                    totalSize, ctx.getDevice().getName(), buffers.size() * entrySize, e);

            for (CudaMemByte buffer: buffers) {
                buffer.free();
            }

            throw e;
        }

        logger.info("allocated {} devices cache slots", numEntries);
    }

    public long getEntrySize() {
        return entrySize;
    }

    public long getNumEntries() {
        return numEntries;
    }

    @Override
    protected Optional<CudaMemByte> createBuffer(String key) {
        return Optional.ofNullable(buffers.poll());
    }

    @Override
    protected void destroyBuffer(CudaMemByte buffer) {
        buffers.push(buffer);
    }

    @Override
    public synchronized void cleanup() {
        super.cleanup();

        while (!buffers.isEmpty()) {
            buffers.pop().free();
        }
    }
}

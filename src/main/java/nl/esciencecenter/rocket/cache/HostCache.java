package nl.esciencecenter.rocket.cache;

import jcuda.CudaException;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaPinned;

import java.util.ArrayDeque;
import java.util.Optional;

import static jcuda.driver.JCudaDriver.CU_MEMHOSTALLOC_PORTABLE;

public class HostCache extends AbstractCache<CudaPinned, Long> {
    private ArrayDeque<CudaPinned> buffers;
    private long numEntries;
    private long entrySize;

    public HostCache(CudaContext ctx, long totalSize, long bufferSize) throws CudaException {
        long n = totalSize / bufferSize;
        buffers = new ArrayDeque<>();
        numEntries = n;
        entrySize = bufferSize;

        try {
            for (int i = 0; i < n; i++) {
                buffers.push(ctx.allocHostBytes((int) bufferSize, CU_MEMHOSTALLOC_PORTABLE));
            }
        } catch (Throwable e) {
            for (CudaPinned buffer: buffers) {
                buffer.free();
            }

            throw e;
        }
    }

    public long getEntrySize() {
        return entrySize;
    }

    public long getNumEntries() {
        return numEntries;
    }

    @Override
    protected Optional<CudaPinned> createBuffer(String key) {
        return Optional.ofNullable(buffers.poll());
    }

    @Override
    protected void destroyBuffer(CudaPinned buffer) {
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

package nl.esciencecenter.phylogenetics_analysis;

import jcuda.Sizeof;
import nl.esciencecenter.phylogenetics_analysis.kernels.CompositionVectorKernels;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaMem;
import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
import nl.esciencecenter.rocket.scheduler.ApplicationContext;
import nl.esciencecenter.xenon.filesystems.Path;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.zip.GZIPInputStream;

public class SequenceAnalysisContext implements ApplicationContext<SequenceIdentifier, Double> {
    protected static final Logger logger = LogManager.getLogger();
    private CompositionVectorKernels kernel;
    private int maxVectorSize;
    private int maxInputSize;
    private CudaMemByte scratch;

    static private final int KEY_SIZE = CompositionVectorKernels.KEY_SIZE;
    static private final int VALUE_SIZE = CompositionVectorKernels.VALUE_SIZE;

    public SequenceAnalysisContext(CudaContext context, String alphabet, int k, int maxVectorSize, int maxInputSize) {
        while (maxVectorSize % 32 != 0) maxVectorSize++;
        kernel = new CompositionVectorKernels(context, alphabet, k, maxVectorSize, maxInputSize);

        this.maxVectorSize = maxVectorSize;
        this.maxInputSize = maxInputSize;
        this.scratch = context.allocBytes(1 );
    }

    public long getMaxFileSize() {
        return maxInputSize * Sizeof.CHAR;
    }

    @Override
    public long getMaxInputSize() {
        return maxVectorSize * (CompositionVectorKernels.KEY_SIZE + CompositionVectorKernels.VALUE_SIZE);
    }

    @Override
    public long getMaxOutputSize() {
        return Sizeof.DOUBLE;
    }

    @Override
    public Path[] getInputFiles(SequenceIdentifier key) {
        return new Path[]{
                new Path(key.getPath())
        };
    }

    @Override
    public long parseFiles(SequenceIdentifier key, ByteBuffer[] inputs, ByteBuffer output) {
        byte[] array = inputs[0].array();
        InputStream stream;

        try {
            stream = new GZIPInputStream(new ByteArrayInputStream(array));
        } catch (Exception e) {
            // ignore exception, probably not a GZIP file
            stream = new ByteArrayInputStream(array);
        }

        try {
            return ReadFasta.read(key.getPath(), stream, output);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public long preprocessInputGPU(SequenceIdentifier key, CudaMem inputBuf, CudaMem outputBuf) {
        CudaMemByte input = inputBuf.asBytes();
        CudaMemByte output = outputBuf.asBytes();

        CudaMemByte outputKeys = output.slice(0, maxVectorSize * KEY_SIZE);
        CudaMemByte outputValues = output.slice(maxVectorSize * KEY_SIZE, maxVectorSize * VALUE_SIZE);

        int size = kernel.buildVectorGPU(
                input,
                outputKeys,
                outputValues
        );

        int offset = 0;
        while (offset < size) {
            int chunk = Math.min(size - offset, (maxVectorSize - size));
            CudaMemByte src = output.slice((maxVectorSize + offset) * KEY_SIZE, chunk * KEY_SIZE);
            CudaMemByte dst = output.slice((size + offset) * KEY_SIZE, chunk * KEY_SIZE);
            src.copyToDevice(dst);

            offset += chunk;
        }

        logger.trace("loaded {} and final result has size {}", key, size);
        return size * (KEY_SIZE + VALUE_SIZE);
    }

    @Override
    public long correlateGPU(SequenceIdentifier leftKey, CudaMem left, SequenceIdentifier rightKey, CudaMem right, CudaMem output) {
        int leftSize = (int)(left.sizeInBytes() / (KEY_SIZE + VALUE_SIZE));
        CudaMemByte leftKeys = left
                .asBytes()
                .slice(0, leftSize * KEY_SIZE);
        CudaMemByte leftValues = left
                .asBytes()
                .slice(leftKeys.elementCount())
                .slice(0, leftSize * VALUE_SIZE);

        int rightSize = (int)(right.sizeInBytes() / (KEY_SIZE + VALUE_SIZE));
        CudaMemByte rightKeys = right
                .asBytes()
                .slice(0, rightSize * KEY_SIZE);
        CudaMemByte rightValues = right
                .asBytes()
                .slice(rightKeys.elementCount())
                .slice(0, rightSize * VALUE_SIZE);

        CudaMemDouble result = output.asDoubles();

        kernel.compareVectorsGPU(leftKeys, leftValues, leftSize, rightKeys, rightValues, rightSize, result);

        return output.sizeInBytes();
    }

    @Override
    public Double postprocessOutput(SequenceIdentifier leftKey, SequenceIdentifier rightKey, ByteBuffer output) {
        return output.asDoubleBuffer().get(0);
    }

    @Override
    public void destroy() {
        scratch.free();
        kernel.cleanup();
    }
}

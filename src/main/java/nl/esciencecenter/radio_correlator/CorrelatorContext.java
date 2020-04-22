package nl.esciencecenter.radio_correlator;

import jcuda.CudaException;
import jcuda.Sizeof;
import nl.esciencecenter.radio_correlator.kernels.Correlator;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaMem;
import nl.esciencecenter.rocket.cubaapi.CudaStream;
import nl.esciencecenter.rocket.scheduler.ApplicationContext;
import nl.esciencecenter.xenon.filesystems.Path;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

public class CorrelatorContext implements ApplicationContext<StationIdentifier, float[]> {
    private CudaContext context;
    private Correlator correlator;
    private CudaStream stream;
    private int numChannels;
    private int numTimes;

    CorrelatorContext(CudaContext context, int numChannels, int numTimes) throws IOException, CudaException {
        this.context = context;
        this.stream = context.createStream();
        this.correlator = new Correlator(context, numChannels, numTimes);
        this.numChannels = numChannels;
        this.numTimes = numTimes;
    }

    @Override
    public long getMaxFileSize() {
        return getMaxInputSize(); // 2*FLOAT => complex number
    }

    @Override
    public long getMaxInputSize() {
        return numChannels * numTimes * Correlator.NUM_POLARIZATIONS * 2 * Sizeof.FLOAT; // 2*FLOAT => complex number
    }

    @Override
    public long getMaxOutputSize() {
        return Correlator.NUM_POLARIZATIONS * Correlator.NUM_POLARIZATIONS * numChannels * 2 * Sizeof.FLOAT; // 2*FLOAT => complex number
    }

    @Override
    public Path[] getInputFiles(StationIdentifier key) {
        return new Path[]{
                new Path(key.getPath())
        };
    }

    @Override
    public long parseFiles(StationIdentifier key, ByteBuffer[] inputs, ByteBuffer output) {
        return 0;
    }

    @Override
    public long preprocessInputGPU(StationIdentifier key, CudaMem input, CudaMem output) {
        output.asFloats().fillAsync(0, stream);
        stream.synchronize();
        return output.sizeInBytes();
    }

    @Override
    public long correlateGPU(
            StationIdentifier leftKey, CudaMem leftMem,
            StationIdentifier rightKey, CudaMem rightMem,
            CudaMem output
    ) {
        correlator.compareGPU(
                leftMem.asFloats(),
                rightMem.asFloats(),
                output.asFloats());
        return output.sizeInBytes();
    }

    @Override
    public float[] postprocessOutput(StationIdentifier left, StationIdentifier right, ByteBuffer buffer) {
        FloatBuffer fb = buffer.asFloatBuffer();
        float[] output = new float[fb.capacity()];
        fb.get(output);
        return output;
    }

    @Override
    public void destroy() {
        stream.destroy();
        correlator.cleanup();
    }
}

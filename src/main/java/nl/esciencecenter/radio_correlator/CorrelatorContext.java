package nl.esciencecenter.radio_correlator;

import jcuda.CudaException;
import jcuda.Sizeof;
import nl.esciencecenter.radio_correlator.kernels.Correlator;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaMem;
import nl.esciencecenter.rocket.cubaapi.CudaStream;
import nl.esciencecenter.rocket.indexspace.CorrelationSpawner;
import nl.esciencecenter.rocket.types.ApplicationContext;
import nl.esciencecenter.rocket.types.InputTask;
import nl.esciencecenter.rocket.types.LeafTask;
import nl.esciencecenter.rocket.util.Correlation;
import nl.esciencecenter.xenon.filesystems.Path;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.List;

public class CorrelatorContext implements ApplicationContext {
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

    public Input getInput(StationIdentifier key) {
        return new Input(key);
    }

    static class Input implements InputTask {
        private StationIdentifier key;

        Input(StationIdentifier key) {
            this.key = key;
        }

        @Override
        public StationIdentifier getKey() {
            return key;
        }

        @Override
        public Path[] getInputs() {
            return new Path[]{
                    new Path(key.getPath())
            };
        }

        @Override
        public long preprocess(ByteBuffer[] inputs, ByteBuffer output) {
            return 0;
        }

        @Override
        public long execute(ApplicationContext context, CudaMem input, CudaMem output) {
            CorrelatorContext c = (CorrelatorContext) context;
            output.asFloats().fillAsync(0, c.stream);
            c.stream.synchronize();
            return output.sizeInBytes();
        }
    }

    static class Task implements LeafTask<Correlation<StationIdentifier, float[]>> {
        private StationIdentifier left;
        private StationIdentifier right;

        Task(StationIdentifier left, StationIdentifier right) {
            this.left = left;
            this.right = right;
        }

        @Override
        public List<InputTask> getInputs() {
            return List.of(new Input(left), new Input(right));
        }

        @Override
        public long execute(
                ApplicationContext context,
                CudaMem[] inputs,
                CudaMem output
        ) {
            CorrelatorContext c = (CorrelatorContext) context;
            c.correlator.compareGPU(
                    inputs[0].asFloats(),
                    inputs[1].asFloats(),
                    output.asFloats());
            return output.sizeInBytes();
        }

        @Override
        public Correlation<StationIdentifier, float[]> postprocess(
                ApplicationContext context,
                ByteBuffer buffer
        ) {
            FloatBuffer fb = buffer.asFloatBuffer();
            float[] output = new float[fb.capacity()];
            fb.get(output);
            return new Correlation<>(left, right, output);
        }

        @Override
        public String toString() {
            return left + " x " + right;
        }
    }


    static class Spawner implements CorrelationSpawner<StationIdentifier, Correlation<StationIdentifier, float[]>> {
        Spawner() {
            //
        }

        @Override
        public LeafTask<Correlation<StationIdentifier, float[]>> spawn(StationIdentifier left, StationIdentifier right) {
            return new Task(left, right);
        }
    }

    @Override
    public void destroy() {
        stream.destroy();
        correlator.cleanup();
    }
}

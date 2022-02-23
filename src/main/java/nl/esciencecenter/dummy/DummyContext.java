package nl.esciencecenter.dummy;


import jcuda.Sizeof;
import nl.esciencecenter.rocket.cubaapi.CudaMem;
import nl.esciencecenter.rocket.indexspace.CorrelationSpawner;
import nl.esciencecenter.rocket.types.ApplicationContext;
import nl.esciencecenter.rocket.types.InputTask;
import nl.esciencecenter.rocket.types.LeafTask;
import nl.esciencecenter.rocket.util.Correlation;
import nl.esciencecenter.xenon.filesystems.Path;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.ByteBuffer;
import java.util.List;


public class DummyContext implements ApplicationContext {
    protected static final Logger logger = LogManager.getLogger();

    public enum ComparisonStrategy {
        PCE,
        NCC
    }

    private int fileSize;
    private int inputSize;

    public DummyContext(int inputSize, int fileSize) {
        this.inputSize = inputSize;
        this.fileSize = fileSize;
    }

    @Override
    public long getMaxFileSize() {
        return fileSize;
    }

    @Override
    public long getMaxInputSize() {
        return inputSize;
    }

    @Override
    public long getMaxOutputSize() {
        return Sizeof.INT;
    }

    public Input getInput(DummyIdentifier key) {
        return new Input(key);
    }

    static class Input implements InputTask {
        private DummyIdentifier id;

        Input(DummyIdentifier id) {
            this.id = id;
        }

        @Override
        public DummyIdentifier getKey() {
            return id;
        }

        @Override
        public Path[] getInputs() {
            return new Path[0];
        }

        @Override
        public long preprocess(ByteBuffer[] inputBuffers, ByteBuffer outputBuffer) {
            return 0;
        }

        @Override
        public long execute(ApplicationContext context, CudaMem input, CudaMem output) {
            DummyContext c = (DummyContext) context;
            return c.getMaxInputSize();
        }
    }

    static class Task implements LeafTask<Correlation<DummyIdentifier, Integer>> {
        private DummyIdentifier left;
        private DummyIdentifier right;

        Task(DummyIdentifier left, DummyIdentifier right) {
            this.left = left;
            this.right = right;
        }

        @Override
        public List<InputTask> getInputs() {
            return List.of(new Input(left), new Input(right));
        }

        @Override
        public long execute(ApplicationContext context, CudaMem[] inputs, CudaMem output) {
            DummyContext c = (DummyContext) context;
            return c.getMaxOutputSize();
        }

        @Override
        public Correlation<DummyIdentifier, Integer> postprocess(ApplicationContext context, ByteBuffer buffer) {
            int result = left.getIndex() + 2 * right.getIndex();
            return new Correlation<>(left, right, result);
        }

        @Override
        public String toString() {
            return left + " x " + right;
        }
    }

    public Task getTask(DummyIdentifier left, DummyIdentifier right) {
        return new Task(left, right);
    }

    static class Spawner implements CorrelationSpawner<DummyIdentifier, Correlation<DummyIdentifier, Integer>> {
        Spawner() {
            //
        }

        @Override
        public LeafTask<Correlation<DummyIdentifier, Integer>> spawn(DummyIdentifier left, DummyIdentifier right) {
            return new Task(left, right);
        }
    }

    @Override
    public void destroy() {
        //
    }
}

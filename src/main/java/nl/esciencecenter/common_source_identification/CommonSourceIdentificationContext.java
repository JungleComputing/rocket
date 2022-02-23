package nl.esciencecenter.common_source_identification;

import jcuda.CudaException;
import jcuda.Sizeof;
import nl.esciencecenter.common_source_identification.kernels.compare.NormalizedCrossCorrelation;
import nl.esciencecenter.common_source_identification.kernels.compare.PatternComparator;
import nl.esciencecenter.common_source_identification.kernels.compare.PeakToCorrelationEnergy;
import nl.esciencecenter.common_source_identification.kernels.filter.PRNUFilter;
import nl.esciencecenter.common_source_identification.util.Dimension;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaMem;
import nl.esciencecenter.rocket.indexspace.CorrelationSpawner;
import nl.esciencecenter.rocket.types.ApplicationContext;
import nl.esciencecenter.rocket.types.InputTask;
import nl.esciencecenter.rocket.types.LeafTask;
import nl.esciencecenter.rocket.util.Correlation;
import nl.esciencecenter.xenon.filesystems.Path;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;

public class CommonSourceIdentificationContext implements ApplicationContext {
    protected static final Logger logger = LogManager.getLogger();

    public enum ComparisonStrategy {
        PCE,
        NCC
    }

    private Dimension imageDim;
    private PRNUFilter filter;
    private PatternComparator pce;

    public CommonSourceIdentificationContext(CudaContext context, Dimension dim, ComparisonStrategy comp) throws CudaException, IOException {
        this.imageDim = dim;

        if (ComparisonStrategy.PCE.equals(comp)) {
            this.filter = new PRNUFilter(
                    dim.getHeight(),
                    dim.getWidth(),
                    context,
                    true);

            this.pce = new PeakToCorrelationEnergy(
                    dim.getHeight(),
                    dim.getWidth(),
                    context,
                    true);
        } else if (ComparisonStrategy.NCC.equals(comp)) {
            this.filter = new PRNUFilter(
                    dim.getHeight(),
                    dim.getWidth(),
                    context,
                    false);

            this.pce = new NormalizedCrossCorrelation(
                    dim.getHeight(),
                    dim.getWidth(),
                    context);
        } else {
            throw new IllegalArgumentException("unknown comparison strategy: " + comp);
        }
    }

    @Override
    public long getMaxFileSize() {
        return filter.getInputSize();
    }

    @Override
    public long getMaxInputSize() {
        return filter.getOutputSize();
    }

    @Override
    public long getMaxOutputSize() {
        return Sizeof.DOUBLE;
    }


    public Input getInput(ImageIdentifier key) {
        return new Input(key, imageDim);
    }

    @Override
    public void destroy() {
        filter.cleanup();
        pce.cleanup();
    }

    static class Input implements InputTask {
        private ImageIdentifier id;
        private Dimension dim;

        Input(ImageIdentifier id, Dimension dim) {
            this.id = id;
            this.dim = dim;
        }

        @Override
        public Path[] getInputs() {
            logger.trace("loading {}", id);

            return new Path[]{
                    new Path(id.getPath())
            };
        }

        public ImageIdentifier getKey() {
            return id;
        }

        @Override
        public long preprocess(ByteBuffer[] inputBuffers, ByteBuffer outputBuffer) {
            Dimension imageDim;

            try {
                imageDim = ReadJPEG.readJPEG(inputBuffers[0], outputBuffer);
            } catch (Exception e) {
                throw new RuntimeException("error while reading: " + id.getPath(), e);
            }

            if (!imageDim.equals(dim)) {
                throw new IllegalArgumentException("image has incorrect dimensions: " + id.getPath());
            }

            return outputBuffer.capacity();
        }

        @Override
        public long execute(ApplicationContext context, CudaMem input, CudaMem output) {
            CommonSourceIdentificationContext c = (CommonSourceIdentificationContext) context;
            c.filter.applyGPU(input, output);
            return c.filter.getOutputSize();
        }
    }

    static class Task implements LeafTask<Correlation<ImageIdentifier, Double>> {
        private ImageIdentifier left;
        private ImageIdentifier right;
        private Dimension dim;

        Task(ImageIdentifier left, ImageIdentifier right, Dimension dim) {
            this.left = left;
            this.right = right;
            this.dim = dim;
        }

        @Override
        public List<InputTask> getInputs() {
            return List.of(new Input(left, dim), new Input(right, dim));
        }


        @Override
        public long execute(ApplicationContext context, CudaMem[] inputs, CudaMem output) {
            logger.trace("correlation {}", this);
            CommonSourceIdentificationContext c = (CommonSourceIdentificationContext) context;
            c.pce.applyGPU(inputs[0], inputs[1], output);
            return  Sizeof.DOUBLE;
        }

        @Override
        public Correlation<ImageIdentifier, Double> postprocess(ApplicationContext context, ByteBuffer buffer) {
            Double corr = (Double) buffer.asDoubleBuffer().get(0);
            return new Correlation<>(left, right, corr);
        }

        @Override
        public String toString() {
            return left + " x " + right;
        }
    }

    static class Spawner implements CorrelationSpawner<ImageIdentifier, Correlation<ImageIdentifier, Double>> {
        private Dimension dim;

        Spawner(Dimension dim) {
            this.dim = dim;
        }

        @Override
        public LeafTask<Correlation<ImageIdentifier, Double>> spawn(ImageIdentifier left, ImageIdentifier right) {
            return new Task(left, right, dim);
        }
    }
}

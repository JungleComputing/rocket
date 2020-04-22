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
import nl.esciencecenter.rocket.scheduler.ApplicationContext;
import nl.esciencecenter.xenon.filesystems.Path;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.ByteBuffer;

public class CommonSourceIdentificationContext implements ApplicationContext<ImageIdentifier, Double> {
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

    @Override
    public Path[] getInputFiles(ImageIdentifier id) {
        logger.trace("loading {}", id);

        return new Path[]{
            new Path(id.getPath())
        };
    }

    @Override
    public long parseFiles(ImageIdentifier id, ByteBuffer[] inputBuffers, ByteBuffer outputBuffer) {
        Dimension imageDim;

        try {
            imageDim = ReadJPEG.readJPEG(inputBuffers[0], outputBuffer);
        } catch (Exception e) {
            throw new RuntimeException("error while reading: " + id.getPath(), e);
        }

        if (!imageDim.equals(this.imageDim)) {
            throw new IllegalArgumentException("image has incorrect dimensions: " + id.getPath());
        }

        return outputBuffer.capacity();
    }

    @Override
    public long preprocessInputGPU(ImageIdentifier id, CudaMem input, CudaMem output) {
        filter.applyGPU(input, output);
        return filter.getOutputSize();
    }

    @Override
    public long correlateGPU(ImageIdentifier left, CudaMem leftMem, ImageIdentifier right, CudaMem rightMem, CudaMem output) {
        logger.trace("correlation {}x{}", left, right);
        pce.applyGPU(leftMem, rightMem, output);
        return  Sizeof.DOUBLE;
    }

    @Override
    public Double postprocessOutput(ImageIdentifier left, ImageIdentifier right, ByteBuffer buffer) {
        return buffer.asDoubleBuffer().get(0);
    }

    @Override
    public void destroy() {
        filter.cleanup();
        pce.cleanup();
    }
}

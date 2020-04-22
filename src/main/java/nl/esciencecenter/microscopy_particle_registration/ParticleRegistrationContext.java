package nl.esciencecenter.microscopy_particle_registration;

import jcuda.Sizeof;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import nl.esciencecenter.microscopy_particle_registration.kernels.PairFitting;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaMem;
import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;
import nl.esciencecenter.rocket.cubaapi.CudaStream;
import nl.esciencecenter.rocket.scheduler.ApplicationContext;
import nl.esciencecenter.microscopy_particle_registration.kernels.expdist.ExpDist;
import nl.esciencecenter.xenon.filesystems.Path;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;

public class ParticleRegistrationContext implements ApplicationContext<ParticleIdentifier, ParticleMatching> {
    protected static final Logger logger = LogManager.getLogger();

    final static private Vec[] INITIAL_GUESSES = new Vec[]{
            DenseVector.toDenseVec(0.0, 0.0, -1.00 * Math.PI),
            DenseVector.toDenseVec(0.0, 0.0, -0.75 * Math.PI),
            DenseVector.toDenseVec(0.0, 0.0, -0.50 * Math.PI),
            DenseVector.toDenseVec(0.0, 0.0, -0.25 * Math.PI),
            DenseVector.toDenseVec(0.0, 0.0, 0.00 * Math.PI),
            DenseVector.toDenseVec(0.0, 0.0, 0.25 * Math.PI),
            DenseVector.toDenseVec(0.0, 0.0, 0.50 * Math.PI),
            DenseVector.toDenseVec(0.0, 0.0, 0.75 * Math.PI),
    };

    private int maxSize;
    private CudaStream stream;
    private PairFitting pairFitting;
    private ExpDist expDist;


    public ParticleRegistrationContext(CudaContext context, PairFitting pairFitting, ExpDist expDist, int maxSize) throws IOException {
        this.pairFitting = pairFitting;
        this.expDist = expDist;

        this.stream = context.createStream();
        this.maxSize = maxSize;
    }

    public long getMaxFileSize() {
        return getMaxInputSize();
    }

    @Override
    public long getMaxInputSize() {
        return maxSize * 3 * Sizeof.DOUBLE; // Input entry is 3 doubles for x,y,z
    }

    @Override
    public long getMaxOutputSize() {
        return 4 * Sizeof.DOUBLE; // Output is (score, translate_x, translate_y, rotation)
    }

    @Override
    public Path[] getInputFiles(ParticleIdentifier key) {
        return new Path[]{
                new Path(key.getPath())
        };
    }

    @Override
    public long parseFiles(ParticleIdentifier key, ByteBuffer[] inputs, ByteBuffer buffer) {
        logger.trace("start parsing {}", key);
        ByteArrayInputStream stream = new ByteArrayInputStream(inputs[0].array());
        JSONArray object = new JSONArray(new JSONTokener(stream));

        int n = object.length();
        double[] pos = new double[2 * n];
        double[] sigma = new double[n];

        // Read records
        for (int i = 0; i < n; i++) {
            JSONObject record = object.getJSONObject(i);
            pos[2 * i + 0] = record.getDouble("x");
            pos[2 * i + 1] = record.getDouble("y");
            sigma[i] = record.getDouble("localization_uncertainty");
        }

        // Write to buffer
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        DoubleBuffer dbuffer = buffer.asDoubleBuffer();
        dbuffer.clear(); // This sets position to zero, does not actually clear anything
        dbuffer.put(pos);
        dbuffer.put(sigma);

        return dbuffer.position() * Sizeof.DOUBLE;
    }

    @Override
    public long preprocessInputGPU(ParticleIdentifier s, CudaMem input, CudaMem output) {
        output.asBytes().copyFromDevice(input.asBytes());
        return input.sizeInBytes();
    }

    @Override
    public long correlateGPU(
            ParticleIdentifier left,
            CudaMem leftMem,
            ParticleIdentifier right,
            CudaMem rightMem,
            CudaMem outputMem
    ) {
        logger.trace("start correlating {} x {}", left, right);
        int m = left.getNumberOfPoints();
        int n = right.getNumberOfPoints();
        CudaMemDouble model = leftMem.asDoubles().slice(0, 2 * m);
        CudaMemDouble scene = rightMem.asDoubles().slice(0, 2 * n);
        CudaMemDouble modelSigmas = leftMem.asDoubles().slice(2 * m, m);
        CudaMemDouble sceneSigmas = rightMem.asDoubles().slice(2 * n, n);

        double bestScore = Double.NEGATIVE_INFINITY;
        Vec bestParam = new DenseVector(3);
        Vec outputParam = new DenseVector(3);

        for (Vec guessParam: INITIAL_GUESSES) {
            pairFitting.applyGPU(
                    model,
                    scene,
                    guessParam.clone(),
                    outputParam);

            double score = expDist.applyGPU(
                    model,
                    scene,
                    outputParam.get(0),
                    outputParam.get(1),
                    outputParam.get(2),
                    modelSigmas,
                    sceneSigmas);

            if (score > bestScore) {
                bestParam = outputParam.clone();
                bestScore = score;
            }
        }

        double[] result = new double[]{
                bestScore,
                bestParam.get(0),
                bestParam.get(1),
                bestParam.get(2)
        };

        outputMem.asDoubles().copyFromHostAsync(result, stream);
        stream.synchronize();
        logger.trace("finish correlating {} x {}", left, right);

        return outputMem.sizeInBytes();
    }

    @Override
    public ParticleMatching postprocessOutput(ParticleIdentifier left, ParticleIdentifier right, ByteBuffer buffer) {
        DoubleBuffer dbuffer = buffer.order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer();

        return new ParticleMatching(
                dbuffer.get(0),
                dbuffer.get(1),
                dbuffer.get(2),
                dbuffer.get(3)
        );
    }

    @Override
    public void destroy() {
        expDist.cleanup();
        pairFitting.cleanup();
        stream.destroy();
    }
}

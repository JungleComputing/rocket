package nl.esciencecenter.rocket.types;

import nl.esciencecenter.rocket.cubaapi.CudaMem;

import java.nio.ByteBuffer;
import java.util.List;

public interface LeafTask<R> {
    /**
     *
     * @return
     */
    public List<InputTask> getInputs();

    /**
     * Calculates the correlation of two keys on the GPU. The output data must be written to the given output buffer.
     * It is not allowed to modify the data in the two input buffers since they will be reused for future calls. The
     * function should return the size of the output buffer, i.e., the number of bytes that will be transferred back
     * to the host. This is useful for variable-length outputs where only the first bytes of the output are used.
     *
     * @param inputs The input buffers.
     * @param output The output buffer.
     * @return The size of the output buffer which is utilized.
     */
    public long execute(ApplicationContext context, CudaMem[] inputs, CudaMem output);

    /**
     * Process the result of a correlation (i.e., the result of execute) and return an object. This object can
     * either be the result of the correlation (e.g., Float object of the correlation score) or some identifier (e.g.,
     * the Path where the result was written to).
     *
     * @param output The output data of execute
     * @return The resulting object.
     */
    public R postprocess(ApplicationContext context, ByteBuffer output);
}

package nl.esciencecenter.rocket.scheduler;

import nl.esciencecenter.rocket.cubaapi.CudaMem;
import nl.esciencecenter.rocket.util.Tuple;
import nl.esciencecenter.xenon.filesystems.Path;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.Optional;

public interface ApplicationContext<K, R> {

    /**
     * Returns the maximum buffer size required to store the output of parseFiles and the input of preprocessInputGPU
     * This size is used to preallocate internal buffers.
     *
     * @return The buffer size;
     */
    public long getMaxFileSize();

    /**
     * Returns the maximum input buffer size required for the preprocessed input. This size is used to
     * preallocate internal buffers.
     *
     * @return The buffer size.
     */
    public long getMaxInputSize();

    /**
     * Returns the maximum output buffer size required for the output of correlateGPU. This size is used to
     * preallocate internal buffers.
     *
     * @return The buffer size.
     */
    public long getMaxOutputSize();

    /**
     * Returns the paths of the input files required for the given key.
     *
     * @return
     */
    public Path[] getInputFiles(K key);

    /**
     * Load the data associated with the given key. The data must written into the given buffer which has a
     * capacity of getMaxFileSize(). The function should return the size of the output buffer, i.e., the number of
     * bytes that will be transferred to the GPU. This is useful for variable-length inputs where only the first bytes
     * are occupied even though the capacity of the buffer is larger.
     *
     * @param key The input key.
     * @param inputs The raw contents of the files provided by getInputFiles(K).
     * @param output The output buffer.
     * @return The size of the output.
     */
    public long parseFiles(K key, ByteBuffer[] inputs, ByteBuffer output);

    /**
     * Performs preprocessing of the input data on the GPU. Data should be read from and written to the given buffer.
     * The capacity of the output is determined by getMaxInputSize().
     *
     * @param key The input key.
     * @param input The input buffer.
     * @param output The output buffer.
     * @return The size of the buffer which is utilized.
     */
    public long preprocessInputGPU(K key, CudaMem input, CudaMem output);

    /**
     * Calculates the correlation of two keys on the GPU. The output data must be written to the given output buffer.
     * It is not allowed to modify the data in the two input buffers since they will be reused for future calls. The
     * function should return the size of the output buffer, i.e., the number of bytes that will be transferred back
     * to the host. This is useful for variable-length outputs where only the first bytes of the output are used.
     *
     * @param leftKey The left key.
     * @param left The left buffer.
     * @param rightKey The right key.
     * @param right The right buffer.
     * @param output The output buffer.
     * @return The size of the output buffer which is utilized.
     */
    public long correlateGPU(K leftKey, CudaMem left, K rightKey, CudaMem right, CudaMem output);

    /**
     * Process the result of a correlation (i.e., the result of correlateGPU) and return an object. This object can
     * either be the result of the correlation (e.g., Float object of the correlation score) or some identifier (e.g.,
     * the Path where the result was written to).
     *
     * @param leftKey The left key.
     * @param rightKey The right key.
     * @param output The output data of correlateGPU
     * @return The resulting object.
     */
    public R postprocessOutput(K leftKey, K rightKey, ByteBuffer output);

    /**
     * Destroy this object (close resources, free memory, etc.)
     */
    public void destroy();
}

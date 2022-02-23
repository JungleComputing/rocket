package nl.esciencecenter.rocket.types;

import nl.esciencecenter.rocket.cubaapi.CudaMem;
import nl.esciencecenter.xenon.filesystems.Path;

import java.nio.ByteBuffer;

public interface InputTask {

    /**
     * Returns the key that uniquely identifies this particular input.
     *
     * @return The key.
     */
    public HashableKey getKey();

    /**
     * Returns the paths of the input files required for the given key.
     *
     * @return The files.
     */
    public Path[] getInputs();

    /**
     * Load the data associated with the given key. The data must written into the given buffer which has a
     * capacity of getMaxFileSize(). The function should return the size of the output buffer, i.e., the number of
     * bytes that will be transferred to the GPU. This is useful for variable-length inputs where only the first bytes
     * are occupied even though the capacity of the buffer is larger.
     *
     * @param inputs The raw contents of the files provided by getInputs().
     * @param output The output buffer.
     * @return The size of the output.
     */
    public long preprocess(ByteBuffer[] inputs, ByteBuffer output);

    /**
     * Performs preprocessing of the input data on the GPU. Data should be read from and written to the given buffer.
     * The capacity of the output is determined by getMaxInputSize().
     *
     * @param input  The input buffer.
     * @param output The output buffer.
     * @return The size of the buffer which is utilized.
     */
    public long execute(ApplicationContext context, CudaMem input, CudaMem output);
}

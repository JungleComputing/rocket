package nl.esciencecenter.rocket.types;

public interface ApplicationContext {

    /**
     * Returns the maximum buffer size required to store the output of preprocess and the input of execute
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
     * Returns the maximum output buffer size required for the output of execute. This size is used to
     * preallocate internal buffers.
     *
     * @return The buffer size.
     */
    public long getMaxOutputSize();


    /**
     * Destroy this object (close resources, free memory, etc.)
     */
    public void destroy();
}

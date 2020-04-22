package nl.esciencecenter.phylogenetics_analysis;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;
import com.sun.jna.Pointer;
import jcuda.Sizeof;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaDevice;
import nl.esciencecenter.rocket.cubaapi.CudaMem;

import java.nio.Buffer;
import java.nio.ByteBuffer;

public interface Bindings extends Library {
    Bindings INSTANCE = (Bindings) Native.load("phylogenetics", Bindings.class);

    long estimateScratchMemory(
            String alphabet,
            int k,
            int max_vector_size);

    int buildCompositionVector(
            Pointer stream,
            Pointer d_temp_storage_ptr,
            long temp_storage_size,
            String alphabet,
            int k,
            Pointer d_string_ptr,
            long string_len,
            Pointer d_vector_keys_ptr,
            Pointer d_vector_values_ptr,
            int[] vector_size_ptr,
            int max_vector_size);

    int compareCompositionVectors(
            Pointer stream,
            Pointer d_temp_storage_ptr,
            long temp_storage_size,
            Pointer d_left_keys_ptr,
            Pointer d_left_values_ptr,
            int left_size,
            Pointer d_right_keys_ptr,
            Pointer d_right_values_ptr,
            int  right_size,
            Pointer d_output_ptr);
}

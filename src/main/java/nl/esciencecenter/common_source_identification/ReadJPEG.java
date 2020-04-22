package nl.esciencecenter.common_source_identification;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import nl.esciencecenter.common_source_identification.util.Dimension;

import java.nio.Buffer;
import java.nio.ByteBuffer;

public class ReadJPEG {
    public interface Bindings extends Library {
        Bindings INSTANCE = (Bindings) Native.load("readjpeg", Bindings.class);

        int readJPEG(
                Buffer input,
                int input_size,
                int[] output_width,
                int[] output_height,
                Buffer output,
                int output_capacity
        );
    }

    public static Dimension readJPEG(
            Buffer input,
            Buffer output
    ) {
        int[] width = new int[1];
        int[] height = new int[1];

        int err = Bindings.INSTANCE.readJPEG(
                input,
                input.remaining(),
                width,
                height,
                output,
                output.remaining());

        if (err != 0) {
            throw new RuntimeException("failed to decode image, native readJPEG returned error code " + err);
        }

        return new Dimension(height[0], width[0]);
    }
}
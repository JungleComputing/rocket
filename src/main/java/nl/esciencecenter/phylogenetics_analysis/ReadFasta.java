package nl.esciencecenter.phylogenetics_analysis;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

public class ReadFasta {
    static long read(String filename, InputStream stream, ByteBuffer output) throws IOException {
        int lineno = 1;
        int length = 0;
        int n = 0;
        byte[] batch = new byte[1024];
        boolean ignoreUntilNewline = false;

        while ((n = stream.read(batch)) > 0) {
            for (int i = 0; i < n; i++) {
                byte b = batch[i];
                if (b == '\n') {
                    lineno++;
                }

                if (ignoreUntilNewline) {
                    ignoreUntilNewline = (b != '\n');
                } else if (b == '#' || b == ';') {
                    // ignore comments
                    ignoreUntilNewline = true;
                } else if (b == '>') {
                    ignoreUntilNewline = true;
                    output.put(length++, (byte) '\n');
                } else if (b == ' ' || b == '\t' || b == '\n') {
                    // ignore whitespace
                } else if (Character.isUpperCase(b)) {
                    output.put(length++, b);
                } else {
                    throw new RuntimeException(String.format(
                            "invalid character '%c' at line %d of file '%s'",
                            (char) b,
                            lineno,
                            filename));
                }

            }
        }

        return length;
    }
}

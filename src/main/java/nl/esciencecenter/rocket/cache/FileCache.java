package nl.esciencecenter.rocket.cache;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

import static java.nio.channels.FileChannel.open;
import static java.nio.file.StandardOpenOption.READ;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;
import static java.nio.file.StandardOpenOption.WRITE;

public class FileCache extends AbstractCache<FileCache.Slot, Long> {
    protected static final Logger logger = LogManager.getLogger();

    static public class Slot {
        private final Path path;

        private Slot(Path path) {
            this.path = path;
        }

        // Read the file and store the result in the given buffer
        public void readSlot(ByteBuffer dst, long n) throws IOException {
            logger.debug("reading {} bytes from {}", n, path);
            dst.position(0);
            dst.limit((int) n);
            long total = 0;
            long i;

            try (FileChannel channel = open(path, READ)) {
                while (total < n) {
                    if ((i = channel.read(dst)) <= 0) {
                        break;
                    }

                    total += i;
                }
            }

            if (total != n) {
                throw new IOException("attempted to read " + n + " bytes, but could only read " + total +
                        " bytes from " + path);
            }
        }

        // Write the given buffer to the file.
        public void writeSlot(ByteBuffer src, long n) throws IOException {
            logger.debug("writing " + n + " bytes to " + path);
            src.position(0);
            src.limit((int) n);
            long total = 0;
            long i;

            try (FileChannel channel = open(path, WRITE, TRUNCATE_EXISTING)) {
                while (total < n) {
                    if ((i = channel.write(src)) <= 0) {
                        break;
                    }

                    total += i;
                }
            }

            if (total != n) {
                throw new IOException("attempted to write " + n + " bytes, but could only write " + total +
                        " bytes to " + path);
            }
        }
    }

    private Path tempDir;
    private int maxFiles;
    private int numFiles;

    public FileCache(Path tempDir, int maxFiles) throws IOException {
        this.tempDir = tempDir.normalize();
        this.maxFiles = maxFiles;
        this.numFiles = 0;


        if (!Files.isDirectory(tempDir)) {
            tempDir = Files.createDirectories(tempDir);
        }

        if (Files.list(tempDir).count() > 0) {
            throw new IOException(tempDir + " is not empty");
        }

        logger.info("File cache enabled using directory {}", tempDir);
    }

    @Override
    protected Optional<Slot> createBuffer(String key) {
        if (numFiles >= maxFiles) {
            return Optional.empty();
        }

        Path path = tempDir.resolve(safeFileName(key));

        try {
            Files.createFile(path);
        } catch (IOException e) {
            logger.warn("failed to create {}: {}", path, e);
            throw new RuntimeException(e);
        }

        numFiles++;
        return Optional.of(new Slot(path));
    }

    @Override
    protected void destroyBuffer(Slot slot) {
        numFiles--;

        try {
            Files.delete(slot.path);
        } catch (IOException e) {
            logger.warn("failed to delete {}: {}", slot.path, e);
        }
    }

    private static String safeFileName(String key) {
        StringBuilder fileName = new StringBuilder();

        for (char c: key.toCharArray()) {
            if ((c >= '0' && c <='9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '.' || c == '_') {
                fileName.append(c);
            } else {
                fileName.append('$');
                fileName.append(Long.toHexString(c));
                fileName.append('$');
            }
        }

        return fileName.toString();
    }
}

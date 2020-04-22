package nl.esciencecenter.common_source_identification;

import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import nl.esciencecenter.common_source_identification.util.Dimension;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.nio.ByteBuffer;

import static java.awt.image.BufferedImage.TYPE_3BYTE_BGR;
import static org.junit.Assert.*;

public class ReadJPEGTest {
    private static BufferedImage generateImage(int width, int height) {
        BufferedImage img = new BufferedImage(width, height, TYPE_3BYTE_BGR);
        byte[] expected = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();

        for (int i = 0; i < expected.length; i++) {
            expected[i] = (byte) i;
        }

        return img;
    }

    private static void testDimension(int width, int height) throws IOException {
        BufferedImage img = generateImage(width, height);

        ByteArrayOutputStream ostream = new ByteArrayOutputStream();
        if (!ImageIO.write(img, "jpeg", ostream )) {
            throw new RuntimeException("failed to write image to JPEG format using ImageIO");
        }

        ByteArrayInputStream istream = new ByteArrayInputStream(ostream.toByteArray());
        img = ImageIO.read(istream);
        if (img == null) {
            throw new RuntimeException("failed to read image from JPEG format using ImageIO");
        }

        ByteBuffer expected = ByteBuffer.wrap(((DataBufferByte) img.getRaster().getDataBuffer()).getData());

        ByteBuffer input = ByteBuffer.wrap(ostream.toByteArray());
        ByteBuffer output = ByteBuffer.allocate(width * height * 3);
        Dimension dim = ReadJPEG.readJPEG(input, output);

        assertEquals(dim, new Dimension(height, width));

        int offset = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int b = 0;  b < 3; b++) {
                    // libjpeg stores pixels in RGB while ImageIo stores them in BGR format.
                    byte expect = expected.get(offset + (2 - b));
                    byte got = output.get(offset + b);

                    if (expect != got) {
                        String msg = String.format("pixel (%d, %d) on channel %d", x, y, b);
                        assertEquals(msg, expect, got);
                    }
                }

                offset += 3;
            }
        }
    }

    @Test
    public void testSmall() throws Throwable {
        testDimension(10, 10);
        testDimension(5, 3);
        testDimension(14, 8);
    }

    @Test
    public void testMedium() throws Throwable {
        testDimension(100, 100);
        testDimension(150, 123);
        testDimension(170, 210);
    }

    @Test
    public void testLarge() throws Throwable {
        testDimension(1000, 1000);
        testDimension(2147, 1523);
        testDimension(2431, 2310);
    }
}
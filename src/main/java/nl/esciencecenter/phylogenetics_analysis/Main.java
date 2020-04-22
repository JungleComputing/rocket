package nl.esciencecenter.phylogenetics_analysis;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import nl.esciencecenter.phylogenetics_analysis.kernels.CompositionVectorKernels;
import nl.esciencecenter.rocket.RocketLauncher;
import nl.esciencecenter.rocket.RocketLauncherArgs;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaDevice;
import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
import nl.esciencecenter.rocket.cubaapi.CudaMemInt;
import nl.esciencecenter.rocket.cubaapi.CudaPinned;
import nl.esciencecenter.rocket.util.Correlation;
import nl.esciencecenter.rocket.util.CorrelationList;
import nl.esciencecenter.rocket.util.FileSystemFactory;
import nl.esciencecenter.rocket.util.Util;
import nl.esciencecenter.xenon.filesystems.FileSystem;
import nl.esciencecenter.xenon.filesystems.Path;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;

import static nl.esciencecenter.rocket.util.Util.stream;

public class Main {
    protected static final Logger logger = LogManager.getLogger();

    public static class MainArgs extends RocketLauncherArgs {
        @Parameter(description="<input dir> <output file>", required = true, arity=2)
        List<String> dirs;

        @Parameter(names={"-k", "--k"}, description="Length of k-mers.")
        int k = 6;

        @Parameter(names="--alphabet", description=".", arity=1)
        String alphabet = "protein";

        @Parameter(names="--find-size", description=".", arity=0)
        boolean findVectorSize = false;

        @Parameter(names="--max-vector-size", description=".", arity=1)
        int maxVectorSize = 0;

        @Parameter(names="--max-input-size", description=".", arity=1)
        int maxInputSize = -1;
    }

    private static void findVectorSizes(FileSystem fs, Path[] files, MainArgs args) {
        CudaContext context = CudaDevice.getBestDevice().createContext();
        int maxVectorSize = 1000 * 1000 * 1000;
        int maxInputSize = maxVectorSize;

        CudaPinned inputHost = context.allocHostBytes(maxInputSize);
        CudaMemByte inputDev = context.allocBytes(maxInputSize);
        CudaMemInt outputKeys = context.allocInts(maxVectorSize);
        CudaMemInt outputValues = context.allocInts(maxVectorSize);

        CompositionVectorKernels kernels = new CompositionVectorKernels(
                context, args.alphabet, args.k, maxVectorSize, maxInputSize);

        long largestSize = 0;
        Path largestSizeFile = files[0];
        long largestN = 0;
        Path largestNFile = files[0];

        for (Path file: files) {
            try (InputStream fileStream = fs.readFromFile(file)) {
                InputStream stream = new BufferedInputStream(fileStream);

                if (file.getFileNameAsString().toLowerCase().endsWith(".gz")) {
                    stream = new GZIPInputStream(stream);
                }

                // Read fasta file
                long size = ReadFasta.read(file.toString(), stream, inputHost.asByteBuffer());

                if (size > largestSize) {
                    largestSizeFile = file;
                    largestSize = size;
                }

                // Copy from host to device
                inputHost.copyToDevice(inputDev, size);

                // Build vector
                long n = kernels.buildVectorGPU(inputDev.slice(0, size), outputKeys, outputValues);

                logger.info("file {}: {} bytes, {} elements", file, size, n);

                if (n > largestN) {
                    largestNFile = file;
                    largestN = n;
                }
            } catch (Exception e) {
                logger.warn("failed to read {}: {}", file, e);
            }
        }

        logger.info("------------------");
        logger.info("largest size {}: {} bytes", largestSizeFile, largestSize);
        logger.info("largest vector {}: {} elements", largestNFile, largestN);
        logger.info("usage: --max-vector-size {} --max-input-size {}", largestN, largestSize);

        inputDev.free();
        inputHost.free();
        outputKeys.free();
        outputValues.free();

        kernels.cleanup();
        context.destroy();
    }


    /**
     * The main routine, it checks the commandline arguments and then calls the non-static onComplete()
     */
    public static void main(final String[] argv) throws Exception {
        MainArgs  args = new MainArgs();
        args.concurrentJobs = 2500;
        args.minimumTileSize = 50;

        try {
            new JCommander(args).parse(argv);
        } catch (ParameterException e) {
            e.getJCommander().getConsole().println(e.getMessage());
            e.getJCommander().usage();
            return;
        }

        String inputDir = args.dirs.get(0);
        String outputFile = args.dirs.get(1);

        FileSystem fs = FileSystemFactory.create(inputDir);

        if (!fs.getAttributes(new Path(".")).isDirectory()) {
            logger.fatal("error: path does not exist or is not a directory: {}", inputDir.toString());
            return;
        }

        Path[] files = Util.scanDirectory(fs, new Path("."), f -> {
            String name = f.toString().toLowerCase();
            return name.endsWith(".faa")
                    || name.endsWith(".faa.gz")
                    || name.endsWith(".fasta")
                    || name.endsWith(".fasta.gz");
        });
        int n = files.length;
        SequenceIdentifier[] identifiers = new SequenceIdentifier[n];

        for (int i = 0; i < identifiers.length; i++) {
            String fileName = fs.getWorkingDirectory().relativize(files[i]).toString();
            identifiers[i] = new SequenceIdentifier(fileName);
        }

        logger.info("found {} available files in {}", n, fs.getWorkingDirectory());

        if (args.findVectorSize) {
            findVectorSizes(fs, files, args);
            return;
        }

        if (args.maxInputSize <= 0) {
            logger.error("No input size given, estimating it to be 1_000_000 (please use --find-size).");
            args.maxInputSize = 1000000;
        }

        if (args.maxVectorSize <= 0) {
            logger.warn("No max vector size given, estimating it to be 1_000_000 (please use --find-size)");
            args.maxVectorSize = 1000000;
        }

        RocketLauncher<SequenceIdentifier, Double> launcher = new RocketLauncher<>(args, fs, context -> {
            return new SequenceAnalysisContext(
                    context,
                    args.alphabet,
                    args.k,
                    args.maxVectorSize,
                    args.maxInputSize);
        });

        CorrelationList<SequenceIdentifier, Double> results = launcher.run(identifiers, true);

        // master only write output
        if (results != null) {
            writeOutput(outputFile, results);
        }
    }

    private static void writeOutput(
            String outputFile,
            CorrelationList<SequenceIdentifier, Double> results
    ) throws IOException {
        HashMap<SequenceIdentifier, Integer> mapping = new HashMap<>();
        JSONArray keys = new JSONArray();
        JSONArray dists = new JSONArray();

        for (Correlation<SequenceIdentifier, Double> r: results) {
            mapping.put(r.getI(), 0);
            mapping.put(r.getJ(), 0);
        }

        int n = 0;
        for (Map.Entry<SequenceIdentifier, Integer> entry: mapping.entrySet()) {
            keys.put(entry.getKey().getPath());
            entry.setValue(n++);
        }

        for (int i = 0; i < n; i++) {
            JSONArray row = new JSONArray();

            for (int j = 0; j < n; j++) {
                row.put(-1);
            }

            dists.put(row);
        }

        for (Correlation<SequenceIdentifier, Double> r: results) {
            int i = mapping.get(r.getI());
            int j = mapping.get(r.getJ());
            double c = r.getCoefficient();

            dists.getJSONArray(i).put(j, c);
            dists.getJSONArray(j).put(i, c);
        }

        JSONObject output = new JSONObject();
        output.put("files", keys);
        output.put("distance", dists);

        try (OutputStream stream = new FileOutputStream(outputFile)) {
            Writer writer = new OutputStreamWriter(new BufferedOutputStream(stream), "UTF-8");
            output.write(writer, 1, 0);
            writer.flush();
        }
    }
}

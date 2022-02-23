package nl.esciencecenter.common_source_identification;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import nl.esciencecenter.common_source_identification.kernels.compare.NormalizedCrossCorrelation;
import nl.esciencecenter.common_source_identification.util.Clustering;
import nl.esciencecenter.common_source_identification.util.Dimension;
import nl.esciencecenter.common_source_identification.util.IO;
import nl.esciencecenter.rocket.RocketLauncher;
import nl.esciencecenter.rocket.RocketLauncherArgs;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaDevice;
import nl.esciencecenter.rocket.cubaapi.CudaMem;
import nl.esciencecenter.rocket.cubaapi.CudaPinned;
import nl.esciencecenter.rocket.util.Correlation;
import nl.esciencecenter.rocket.util.FileSystemFactory;
import nl.esciencecenter.rocket.util.Util;
import nl.esciencecenter.xenon.filesystems.FileSystem;
import nl.esciencecenter.xenon.filesystems.Path;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static nl.esciencecenter.rocket.util.Util.stream;

public class Main {
    protected static final Logger logger = LogManager.getLogger();

    public static class MainArgs {
        @Parameter(description="<output name> <input directory>", required = true, arity=2)
        List<String> dirs;

        @Parameter(names="--ncc", description="Use NCC correlation strategy instead of PCE.")
        boolean useNCC = false;

        @Parameter(names="--tune", description="Tune the correlations kernels.")
        boolean tune = false;
    }

    private static void performTuning(
            FileSystem fs,
            ImageIdentifier[] identifiers,
            Dimension dim,
            CommonSourceIdentificationContext.ComparisonStrategy strategy
    ) throws Exception {
        if (identifiers.length > 2) {
            identifiers = Arrays.copyOf(identifiers, 2);
        }

        CudaContext ctx = CudaDevice.getBestDevice().createContext();
        CommonSourceIdentificationContext f = new CommonSourceIdentificationContext(ctx, dim, strategy);

        logger.info("found device: {}", ctx.getDevice().getName());
        List<CudaMem> patterns = new ArrayList<CudaMem>();

        for (ImageIdentifier id: identifiers) {
            logger.info("loading {}", id);

            CommonSourceIdentificationContext.Input task = f.getInput(id);
            Path[] files = task.getInputs();
            ByteBuffer[] buffers = new ByteBuffer[files.length];

            for (int i = 0; i < files.length; i++) {
                try (InputStream stream = Util.readFile(fs, files[i])) {
                    byte[] buffer = stream.readAllBytes();
                    buffers[i] = ByteBuffer.wrap(buffer);
                }
            }

            CudaPinned pinned = ctx.allocHostBytes(f.getMaxFileSize());
            task.preprocess(buffers, pinned.asByteBuffer());

            CudaMem pattern = ctx.allocBytes(f.getMaxInputSize());
            task.execute(f, pinned.asCudaMem(), pattern);
            ctx.synchronize();

            patterns.add(pattern);
        }


        class Parameter {
            int threadsPerBlock;
            int reducingThreads;

            public Parameter(int threadsPerBlock, int reducingThreads) {
                this.threadsPerBlock = threadsPerBlock;
                this.reducingThreads = reducingThreads;
            }
        }

        List<Parameter> pspace = new ArrayList<>();
        for (int threadsPerBlock: new int[]{64, 128, 256, 512}) {
            for (int reducingThreads: new int[]{64, 128, 256, 512, 1024}) {
                //pspace.add(new Parameter(threadsPerBlock, reducingThreads));
            }
        }

        pspace.add(new Parameter(128, 256));

        Random random = new Random();
        CudaMem output = ctx.allocBytes(f.getMaxOutputSize());

        for (Parameter param: pspace) {
            NormalizedCrossCorrelation ncc = new NormalizedCrossCorrelation(
                    dim.getHeight(),
                    dim.getWidth(),
                    param.threadsPerBlock,
                    param.reducingThreads,
                    ctx);

            // Just to get the JIT going
            ncc.applyGPU(patterns.get(0), patterns.get(0), output);
            ctx.synchronize();

            long before = System.nanoTime();
            int maxTries = 25;
            for (int tries = 0; tries < maxTries; tries++) {
                int i, j;

                do {
                    i = random.nextInt(identifiers.length);
                    j = random.nextInt(identifiers.length);
                } while (i == j);

                ncc.applyGPU(patterns.get(i), patterns.get(j), output);
                ctx.synchronize();
            }
            long after = System.nanoTime();


            ncc.cleanup();

            double averageTime = (after - before) * 1e-9 / maxTries;
            double sizeInGb = (patterns.get(0).sizeInBytes() * 2) * 1e-9;
            double bandwidth = sizeInGb / averageTime;


            logger.info("threads: {}, blocks: {}, time: {} s, bandwidth: {} GB/s",
                    param.threadsPerBlock,
                    param.reducingThreads,
                    averageTime * 1000,
                    bandwidth);

        }
    }

    public static void main(final String[] argv) throws Exception {
        MainArgs args = new MainArgs();
        RocketLauncherArgs largs = new RocketLauncherArgs();
        largs.concurrentJobs = 0;
        largs.minimumTileSize = 0;

        JCommander jcmd = new JCommander();
        jcmd.addObject(args);
        jcmd.addObject(largs);

        try {
            jcmd.parse(argv);
        } catch (ParameterException e) {
            jcmd.getConsole().println(e.getMessage());
            jcmd.usage();
            return;
        }

        CommonSourceIdentificationContext.ComparisonStrategy strategy = args.useNCC
                ? CommonSourceIdentificationContext.ComparisonStrategy.NCC
                : CommonSourceIdentificationContext.ComparisonStrategy.PCE;

        if (largs.concurrentJobs == 0) {
            largs.concurrentJobs = args.useNCC ? 5000 : 250;
        }

        if (largs.minimumTileSize == 0) {
            largs.minimumTileSize = args.useNCC ? 50 : 8;
        }


        String testcase = args.dirs.get(0);
        if (!testcase.matches("[a-zA-Z0-9_.-]+")) {
            System.out.println("testcase will be used for file names, please do not use special characters");
            return;
        }

        String EDGELIST_FILENAME = "edgelist-" + testcase + ".txt";
        String MATRIX_BIN_FILENAME = "matrix-" + testcase + ".dat";
        String MATRIX_TXT_FILENAME = "matrix-" + testcase + ".txt";
        String LINKAGE_FILENAME = "linkage-" + testcase + ".txt";
        String CLUSTERING_FILENAME = "clustering-" + testcase + ".txt";

        FileSystem fs = FileSystemFactory.create(args.dirs.get(1));

        Path inputDir = fs.getWorkingDirectory();
        if (!fs.getAttributes(inputDir).isDirectory()) {
            System.out.println("folderpath does not exist or is not a directory: " + inputDir.toString());
            return;
        }

        // Prepare image;
        Path[] files = Util.scanDirectory(fs, inputDir, f -> f.toString().toLowerCase().endsWith(".jpg"));
        int n = files.length;

        System.out.printf("found %d files in directory %s\n", n, inputDir);

        String[] filenames = new String[n];
        ImageIdentifier[] identifiers = new ImageIdentifier[n];
        for (int i = 0; i < n; i++) {
            filenames[i] = fs.getWorkingDirectory().relativize(files[i]).toString();
            identifiers[i] = new ImageIdentifier(i, filenames[i]);
        }

        // Obtain image dimensions by reading one image
        Dimension dim;
        try (InputStream stream = Util.readFile(fs, files[0])) {
            byte[] input = stream.readAllBytes();
            byte[] output = new byte[3 * 5000 * 5000];

            dim = ReadJPEG.readJPEG(ByteBuffer.wrap(input), ByteBuffer.wrap(output));
        }

        if (args.tune) {
            performTuning(fs, identifiers, dim, strategy);
            return;
        }

        // Launch applications
        RocketLauncher<Correlation<ImageIdentifier, Double>> launcher = new RocketLauncher<>(largs, fs, context -> {
            return new CommonSourceIdentificationContext(context, dim, strategy);
        });

        List<Correlation<ImageIdentifier, Double>> result = launcher.run(
                identifiers,
                false,
                new CommonSourceIdentificationContext.Spawner(dim)
        );

        // Master only
        if (result != null) {
            double[][] corrs = new double[n][n];
            for (Correlation<ImageIdentifier, Double> c: result) {
                int i = c.getI().getIndex();
                int j = c.getJ().getIndex();
                corrs[i][j] = corrs[j][i] = c.getCoefficient();
            }

            //this is where we compute the hierarchical clustering
            ArrayList<Clustering.Link> linkage = Clustering.hierarchical_clustering(corrs, filenames);

            IO.write_linkage(linkage, LINKAGE_FILENAME);
            IO.write_flat_clustering(linkage, filenames, CLUSTERING_FILENAME);

            //write edgelist
            IO.write_edgelist(corrs, filenames, EDGELIST_FILENAME);

            //write the correlation matrix to disk in binary and text form
            IO.write_matrix_text(corrs, MATRIX_TXT_FILENAME);
            IO.write_matrix_binary(corrs, MATRIX_BIN_FILENAME);
        }
    }
}

package nl.esciencecenter.microscopy_particle_registration;


import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import nl.esciencecenter.microscopy_particle_registration.kernels.PairFitting;
import nl.esciencecenter.microscopy_particle_registration.kernels.expdist.ExpDist;
import nl.esciencecenter.rocket.RocketLauncher;
import nl.esciencecenter.rocket.RocketLauncherArgs;
import nl.esciencecenter.rocket.util.Correlation;
import nl.esciencecenter.rocket.util.FileSystemFactory;
import nl.esciencecenter.rocket.util.Util;
import nl.esciencecenter.xenon.filesystems.FileSystem;
import nl.esciencecenter.xenon.filesystems.Path;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;

import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static nl.esciencecenter.rocket.util.Util.stream;

public class Main {
    protected static final Logger logger = LogManager.getLogger();

    public static class MainArgs {
        @Parameter(description="<input dir> <output file>", required = true, arity=2)
        List<String> dirs;

        @Parameter(names="--scale", description="Scale used for calculating Gaussian distance.")
        double scale = 1.0;

        @Parameter(names="--tolerance", description="Tolerance used for optimization function.")
        double tolerance = 1e-15;

        @Parameter(names="--iterations", description="Maximum iterations used for optimization function.")
        int maxIterations = 500;
    }

    /**
     * The main routine, it checks the commandline arguments and then calls the non-static launch()
     */
    public static void main(final String[] argv) throws Exception {
        MainArgs args = new MainArgs();
        RocketLauncherArgs largs = new RocketLauncherArgs();
        largs.concurrentJobs = 10;
        largs.minimumTileSize = 1;

        try {
            JCommander.newBuilder()
                    .addObject(args)
                    .addObject(largs)
                    .build()
                    .parse(argv);
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

        // Prepare files;
        Path[] files = Util.scanDirectory(fs, new Path("."), f -> f.toString().endsWith(".json"));
        int n = files.length;

        String[] fileNames = new String[n];
        ParticleIdentifier[] identifiers = new ParticleIdentifier[n];

        for (int i = 0; i < n; i++) {
            int numPoints;

            try (InputStream stream = Util.readFile(fs, files[i])) {
                JSONArray array = new JSONArray(new JSONTokener(stream));
                numPoints = array.length();
            }

            String path = fs.getWorkingDirectory().relativize(files[i]).toString();
            fileNames[i] = path;
            identifiers[i] = new ParticleIdentifier(i, path, numPoints);
        }

        logger.info("found {} available files in {}", n, inputDir);

        // Launch applications
        double scale = args.scale;
        int maxSize = Arrays
                .stream(identifiers)
                .map(ParticleIdentifier::getNumberOfPoints)
                .reduce(0, Math::max);

        RocketLauncher<Correlation<ParticleIdentifier, ParticleMatching>> launcher = new RocketLauncher<>(largs, fs, context -> {
            return new ParticleRegistrationContext(
                    context,
                    new PairFitting(context, args.scale, args.tolerance, args.maxIterations, maxSize),
                    new ExpDist(context, maxSize),
                    maxSize);
        });

        List<Correlation<ParticleIdentifier, ParticleMatching>> results = launcher.run(
                identifiers,
                true,
                new ParticleRegistrationContext.Spawner()
        );

        // master only write output
        if (results != null) {
            logger.info("writing output to {}", outputFile);

            JSONArray array = new JSONArray();
            List<Correlation<ParticleIdentifier, ParticleMatching>> sorted = new ArrayList<>(results);
            Collections.sort(sorted);

            for (Correlation<ParticleIdentifier, ParticleMatching> c: sorted) {
                String i = c.getI().getPath();
                String j = c.getJ().getPath();
                ParticleMatching m = c.getCoefficient();

                JSONObject obj = new JSONObject();
                obj.put("model", i);
                obj.put("scene", j);
                obj.put("score", m.score);
                obj.put("translate_x", m.translate_x);
                obj.put("translate_y", m.translate_y);
                obj.put("rotate", m.rotation);
                array.put(obj);
            }

            try (OutputStream stream = new FileOutputStream(outputFile)) {
                array.write(new OutputStreamWriter(stream, "UTF-8"));
                stream.flush();
            }
        }
    }
}

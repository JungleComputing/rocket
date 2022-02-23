package nl.esciencecenter.dummy;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import nl.esciencecenter.rocket.RocketLauncher;
import nl.esciencecenter.rocket.RocketLauncherArgs;
import nl.esciencecenter.rocket.util.Correlation;
import nl.esciencecenter.rocket.util.FileSystemFactory;
import nl.esciencecenter.xenon.filesystems.FileSystem;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.ByteBuffer;
import java.util.List;

public class Main {
    protected static final Logger logger = LogManager.getLogger();

    public static class MainArgs {
        @Parameter(names="--inputs", description="Number of inputs", required = true)
        int numInputs;

        @Parameter(names="--input-size", description="Simulated input size in bytes", required = true)
        int inputSize;

        @Parameter(names="--file-size", description="Simulated file size in bytes", required = true)
        int fileSize;

    }


    public static void main(final String[] argv) throws Exception {
        MainArgs args = new MainArgs();
        RocketLauncherArgs largs = new RocketLauncherArgs();

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

        // Create identifiers
        DummyIdentifier[] identifiers = new DummyIdentifier[args.numInputs];
        for (int i = 0; i < identifiers.length; i++) {
            identifiers[i] = new DummyIdentifier(i);
        }

        DummyContext context = new DummyContext(args.inputSize, args.fileSize);
        FileSystem fs = FileSystemFactory.create("/");

        // Launch applications
        RocketLauncher<Correlation<DummyIdentifier, Integer>> launcher = new RocketLauncher<>(largs, fs, ctx -> {
            return context;
        });

        List<Correlation<DummyIdentifier, Integer>> result = launcher.run(
                identifiers,
                false,
                new DummyContext.Spawner()
        );

        // Master only
        if (result != null) {
            for (Correlation<DummyIdentifier, Integer> c: result) {
                int got = c.getCoefficient();
                int expect = context.getTask(c.getI(), c.getJ())
                        .postprocess(context, ByteBuffer.allocate(0))
                        .getCoefficient();

                if (got != expect) {
                    logger.warn("invalid output for ({}, {}): got {} but expected {}",
                            c.getI(), c.getJ(), got, expect);
                }
            }
        }
    }
}

package nl.esciencecenter.radio_correlator;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import nl.esciencecenter.rocket.RocketLauncher;
import nl.esciencecenter.rocket.RocketLauncherArgs;
import nl.esciencecenter.rocket.util.FileSystemFactory;
import nl.esciencecenter.xenon.filesystems.FileSystem;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.util.List;

public class Main {
    protected static final Logger logger = LogManager.getLogger();

    public static class MainArgs {
        @Parameter(description="<input dir> <output dir>", required = true, arity=2)
        List<String> dirs;
    }

    public static void main(final String[] argv) throws Exception {
        MainArgs args = new MainArgs();
        RocketLauncherArgs largs = new RocketLauncherArgs();
        JCommander jcmd = new JCommander(args);
        jcmd.addObject(largs);

        try {
            jcmd.parse(argv);
        } catch (ParameterException e) {
            jcmd.getConsole().println(e.getMessage());
            jcmd.usage();
            return;
        }

        File f = new File(args.dirs.get(0));
        if (!f.exists() || !f.isDirectory()) {
            logger.fatal("error: path does not exist or is not a directory: {}", f.toString());
            return;
        }

        FileSystem fs = FileSystemFactory.create("file:.");

        int numChannels = 256;
        int numTimes = 786;
        int numStations = 1500;
        StationIdentifier[] identifiers = new StationIdentifier[numStations];

        for (int i = 0; i < numStations; i++) {
            identifiers[i] = new StationIdentifier(String.valueOf(i));
        }

        RocketLauncher<StationIdentifier, float[]> launcher = new RocketLauncher<>(largs, fs, context -> {
             return new CorrelatorContext(context, numChannels, numTimes);
        });
    }
}

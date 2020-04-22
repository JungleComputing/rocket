package nl.esciencecenter;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;

public class Main {
    public static void printUsage() {
        System.out.println("Usage: [PROGRAM] [OPTION]...");
        System.out.println(" Where PROGRAM is either 'microscopy', 'radio', 'csi', or 'genetics");
    }

    public static void main(final String[] argv) throws Exception {
        if (argv.length == 0) {
            printUsage();
            return;
        }

        String head = argv[0];
        String[] tail = Arrays.copyOfRange(argv, 1, argv.length);

        if (head.equals("m") || head.equals("microscopy")) {
            nl.esciencecenter.microscopy_particle_registration.Main.main(tail);
        } else if (head.equals("r") || head.equals("radio")) {
            nl.esciencecenter.radio_correlator.Main.main(tail);
        } else if (head.equals("c") || head.equals("csi")) {
            nl.esciencecenter.common_source_identification.Main.main(tail);
        } else if (head.equals("g") || head.equals("genetics")) {
            nl.esciencecenter.phylogenetics_analysis.Main.main(tail);
        } else {
            printUsage();
        }
    }
}

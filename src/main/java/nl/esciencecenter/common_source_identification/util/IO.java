package nl.esciencecenter.common_source_identification.util;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class IO {
    /**
     * This method writes a correlation matrix to a text file
     * The location of the text file is determined by the name of the testcase set by the user
     *
     * @param cortable a double matrix, withSupplier equal width and height, storing the results of a computation of a similarity metric or correlation
     */
    public static void write_matrix_text(double[][] cortable, String MATRIX_TXT_FILENAME) {
        int numfiles = cortable[0].length;
        try {
            PrintWriter textfile = new PrintWriter(MATRIX_TXT_FILENAME);
            for (int i=0; i<numfiles; i++) {
                for (int j=0; j<numfiles; j++) {
                    textfile.format("%.6f, ", cortable[i][j]);
                }
                textfile.println();
            }
            textfile.println();
            textfile.close();
        } catch (Exception ex) {
            System.err.println(ex.getMessage());
        }
    }

    /**
     *
     */
    public static void write_linkage(ArrayList<Clustering.Link> linkage, String LINKAGE_FILENAME) {
        try {
            PrintWriter textfile = new PrintWriter(LINKAGE_FILENAME);
            for (Clustering.Link l : linkage) {
                textfile.println("[" + l.n1 + "," + l.n2 + "," + l.dist + "," + l.size + "]");
            }
            textfile.println();
            textfile.close();
        } catch (Exception ex) {
            System.err.println(ex.getMessage());
        }
    }

    public static void write_flat_clustering(ArrayList<Clustering.Link> linkage, String[] filenames, String CLUSTERING_FILENAME) {
        int N = filenames.length;
        final double THRESHOLD = 60.0;

        try {
            PrintWriter textfile = new PrintWriter(CLUSTERING_FILENAME);
            textfile.println("flat clustering:");

            //create data structures to hold info about clusters
            int next_cluster_id = N;
            HashMap<Integer,ArrayList<Integer>> cluster_members = new HashMap<Integer,ArrayList<Integer>>();
            for (int i=0; i<N; i++) {
                ArrayList<Integer> l = new ArrayList<Integer>(1);
                l.add(i);
                cluster_members.put(i, l);
            }

            boolean termination = false;
            Iterator<Clustering.Link> link_iterator = linkage.iterator();
            for (int i=0; i<N-1 && termination == false; i++) {
                Clustering.Link link = link_iterator.next();
                //System.out.println("[" + link.n1 + "," + link.n2 + "," + link.dist + "," + link.size + "]");

                if (link.dist < THRESHOLD) {
                    for (Map.Entry<Integer, ArrayList<Integer>> entry : cluster_members.entrySet()) {
                        ArrayList<Integer> list = entry.getValue();
                        Collections.sort(list);
                        textfile.println(entry.getKey() + "=" + list.toString());
                    }
                    termination = true;
                }

                if (termination == false) {
                    //merge the clusters into a new cluster in our bookkeeping data structures
                    int cluster1 = link.n1;
                    int cluster2 = link.n2;
                    ArrayList<Integer> cluster1_members = cluster_members.get(cluster1);
                    cluster_members.remove(cluster1);
                    ArrayList<Integer> cluster2_members = cluster_members.get(cluster2);
                    cluster_members.remove(cluster2);
                    cluster1_members.addAll(cluster2_members);
                    cluster_members.put(next_cluster_id, cluster1_members);
                    next_cluster_id += 1;
                }
            }

            textfile.println();
            textfile.flush();
            textfile.println("labels:");

            int[] labeling = new int[N];
            for (int i=0; i<N; i++) {
                labeling[i] = 0;
            }

            int label = 1;
            for (Map.Entry<Integer, ArrayList<Integer>> entry : cluster_members.entrySet()) {
                //System.out.println("label=" + label + "key=" + entry.getKey() + "value=" + entry.getValue().toString() );
                for (Integer m: entry.getValue()) {
                    labeling[m.intValue()] = label;
                }
                label += 1;
            }

            int num_digits = (int)Math.log10(label)+1;
            String format = "%"+num_digits+"d ";
            textfile.print("[");
            for (int l: labeling) {
                textfile.format(format, l);
            }
            textfile.println("]");

            textfile.println();
            textfile.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    /**
     * This method writes a PRNU pattern to a file on disk
     *
     * This method is now only used for debugging purposes because it is much
     * faster to read the JPEG and recompute the PRNU pattern than reading a
     * stored pattern from disk.
     *
     * @param array     a float array containing the PRNU pattern
     * @param filename  a string containing the name of the JPG file, its current extension will be replaced withSupplier '.dat'
     * @param size      the size of the PRNU pattern
     */
    static void write_float_array_to_file(float[] array, String filename, int size, String TEMP_DIR) {
        String file = TEMP_DIR + filename.substring(0, filename.lastIndexOf('.')) + ".dat";

        try {
            DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)));

            for (int i=0;i<size;i++) {
                out.writeFloat(array[i]);
            }
            out.close();
        }
        catch (IOException ex) {
            System.err.println(ex.getMessage());
        }
    }

    /**
     * This method reads a PRNU pattern from a file on disk
     *
     * This method is now only used for debugging purposes because it is much
     * faster to read the JPEG and recompute the PRNU pattern than reading a
     * stored pattern from disk.
     *
     * @param filename  the name of the JPEG file whose PRNU pattern we are now fetching from disk
     * @param size      the size of the PRNU pattern in the number of floats
     * @returns         a float array containing the PRNU pattern
     */
    static float[] read_float_array_from_file(String filename, int size, String TEMP_DIR) {
        String file = TEMP_DIR + '/' + filename.substring(0, filename.lastIndexOf('.')) + ".dat";

        try{
            DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));
            float [] result = new float[size];
            for (int i=0;i<size;i++) {
                result[i] = in.readFloat();
            }
            return result;
        }
        catch (IOException ex) {
            System.err.println(ex.getMessage());
        }
        return null;
    }

    /**
     * This method writes a correlation matrix to an edgelist text file
     * The location of the text file is determined by the name of the testcase set by the user
     *
     * @param cortable      a double matrix, withSupplier equal width and height, storing the results of a computation of a similarity metric or correlation
     * @param filenames     a String array containing the filenames that were compared in the correlation
     */
    public static void write_edgelist(double[][] cortable, String[] filenames, String EDGELIST_FILENAME) {
        PrintWriter edgefile = null;
        try {
            edgefile = new PrintWriter(EDGELIST_FILENAME);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        int n = cortable[0].length;
        for (int i=0; i<n; i++) {
            for (int j=0; j<i; j++) {
                 edgefile.println(filenames[i] + " " + filenames[j] + " " + cortable[i][j]);
            }
        }
        edgefile.close();
    }

    /**
     * This method writes a correlation matrix to a binary file
     * The location of the text file is determined by the name of the testcase set by the user
     * Note that Java writes its doubles in big endian
     *
     * @param cortable a double matrix, withSupplier equal width and height, storing the results of a computation of a similarity metric or correlation
     */
    public static void write_matrix_binary(double[][] cortable, String MATRIX_BIN_FILENAME) {
        int numfiles = cortable[0].length;
        try{
            FileOutputStream fos = new FileOutputStream(MATRIX_BIN_FILENAME);
            DataOutputStream dos = new DataOutputStream(fos);
            for (int i=0; i<numfiles; i++) {
                for (int j=0; j<numfiles; j++) {
                     dos.writeDouble(cortable[i][j]);
                }
            }
        } catch (Exception ex) {
            System.err.println(ex.getMessage());
        }
    }
}

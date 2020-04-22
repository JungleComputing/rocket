package nl.esciencecenter.common_source_identification.util;

import java.util.ArrayList;
import java.util.HashMap;

public class Clustering {
    /**
     *
     */
    public static ArrayList<Link> hierarchical_clustering(double[][] cortable, String[] filenames) {
        int N = filenames.length;
        double[][] matrix = new double[N][N];

        int c = 0;

        //copy cortable into matrix
        for (int i=0; i<N; i++) {
            for (int j=0; j<N; j++) {
                matrix[i][j] = cortable[i][j];
            }
        }

        //create data structures to hold info about clusters
        int next_cluster_id = N;
        ArrayList<Integer> cluster_ids = new ArrayList<Integer>(N);
        for (int i=0; i<N; i++) {
            cluster_ids.add(i,i);
        }
        HashMap<Integer,ArrayList<Integer>> cluster_members = new HashMap<Integer,ArrayList<Integer>>();
        for (int i=0; i<N; i++) {
            ArrayList<Integer> l = new ArrayList<Integer>(1);
            l.add(i);
            cluster_members.put(i, l);
        }

        ArrayList<Link> linkage = new ArrayList<Link>(N-1);

        for (int iterator=0; iterator<N-1; iterator++) {

            //find the most similar pair of clusters
            int[] index_max = findMax(matrix);
            int n1 = index_max[0];
            int n2 = index_max[1];

            if (n1 == n2) {
                break;
            }

            //merge the clusters into a new cluster in our bookkeeping data structures
            int cluster1 = cluster_ids.get(n1);
            int cluster2 = cluster_ids.get(n2);
            ArrayList<Integer> cluster1_members = cluster_members.get(cluster1);
            cluster_members.remove(cluster1);
            ArrayList<Integer> cluster2_members = cluster_members.get(cluster2);
            cluster_members.remove(cluster2);
            cluster1_members.addAll(cluster2_members);
            cluster_members.put(next_cluster_id, cluster1_members);
            cluster_ids.set(n1, next_cluster_id);

            //add to linkage
            int new_size = cluster_members.get(next_cluster_id).size();
            linkage.add(new Link(cluster1, cluster2, matrix[n1][n2], new_size));
            if (new_size >= N) {
                break;
            }

            //update the similarity matrix
            for (int i=0; i<N; i++) {
                if (cluster_members.containsKey(i)) {
                    int other = cluster_ids.get(i);
                    double sum = 0.0;
                    ArrayList<Integer> a = cluster_members.get(next_cluster_id);
                    ArrayList<Integer> b = cluster_members.get(other);

                    for (int j=0; j<a.size(); j++) {
                        for (int k=0; k<b.size(); k++) {
                            sum += cortable[a.get(j)][b.get(k)]; //needs to be cortable NOT matrix
                        }
                    }

                    double avg = sum / (a.size()*b.size());

                    matrix[n1][i] = avg;
                    matrix[i][n1] = avg;
                }
            }

            //erase cluster n2
            for (int i=0; i<N; i++) {
                matrix[n2][i] = -1e200;
                matrix[i][n2] = -1e200;
            }

            //increment next cluster id for next cluster merger
            next_cluster_id += 1;

        }

        return linkage;
    }

    /**
     *
     */
    public static int[] findMax(double[][] cortable) {
        int N = cortable[0].length;
        double highest = -1e100;
        int index[] = new int[2];
        for (int i=0; i<N; i++) {
            for (int j=0; j<N; j++) {
                if (cortable[i][j] > highest) {
                    highest = cortable[i][j];
                    index[0] = i;
                    index[1] = j;
                }
            }
        }
        return index;
    }

    public static class Link {
        public int n1,n2;
        public double dist;
        public int size;
        public Link(int n1, int n2, double dist, int size) {
            this.n1 = n1;
            this.n2 = n2;
            this.dist = dist;
            this.size = size;
        }
    }
}

package nl.esciencecenter.microscopy_particle_registration.kernels.resamplecloud2d;

import nl.esciencecenter.rocket.cubaapi.CudaMemDouble;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

public class ResampleCloud2D {


    public void applyGPU(
            CudaMemDouble pos,
            CudaMemDouble sigma,
            CudaMemDouble resampledPos,
            CudaMemDouble resampledSigma,
            int cutoff
    ) {
        throw new UnsupportedOperationException();
    }

    public void applyCPU(
            double[] pos,
            double[] sigmas,
            double[] resampledPos,
            double[] resampledSigma,
            int cutoff
    ) {
        int n = sigmas.length;
        if (pos.length != 2 * n || resampledPos.length != 2 * n || resampledSigma.length != n) {
            throw new IllegalArgumentException("invalid buffer size");
        }

        // meansigma = mean((Particle.sigma));         % the average uncertainties
        //        xmax=max(Particle.points(:,1));
        //        xmin=min(Particle.points(:,1));
        //        ymax=max(Particle.points(:,2));
        //        ymin=min(Particle.points(:,2));
        double meansigma = 0.0;
        double xmax = Double.NEGATIVE_INFINITY;
        double xmin = Double.POSITIVE_INFINITY;
        double ymax = Double.NEGATIVE_INFINITY;
        double ymin = Double.POSITIVE_INFINITY;

        for (int i = 0; i < n; i++) {
            meansigma += sigmas[i] / n;
            xmin = Math.min(xmin, pos[2 * i + 0]);
            ymin = Math.min(ymin, pos[2 * i + 1]);
            xmax = Math.max(xmax, pos[2 * i + 0]);
            ymax = Math.max(ymax, pos[2 * i + 1]);
        }

        //    % binning the localizations
        //        dmax = [xmax ymax];
        //        dmin = [xmin ymin];
        //        nn = 100;                                   % number of bins
        //        bins = [1 1].*nn;
        //        binsize = (dmax - dmin)./bins;              % bin size
        int nn = 100;
        double binsize = Math.max(xmax - xmin, ymax - ymin) / nn;


        //    % act like addgaussianblob
        //        fsize = double(meansigma./binsize(1));  % filter size
        double fsize = meansigma / binsize;


        //        xi = linspace(dmin(1),dmax(1),bins(1));
        //        yi = linspace(dmin(2),dmax(2),bins(2));
        //
        //        xr = interp1(xi,1:numel(xi),Particle.points(:,1),'nearest');
        //        yr = interp1(yi,1:numel(yi),Particle.points(:,2),'nearest');
        //
        //        subs = [xr yr];
        int[] xi = new int[n];
        int[] yi = new int[n];

        for (int i = 0; i < n; i++) {
            double x = pos[2 * i + 0];
            double y = pos[2 * i + 1];

            int xr = (int)Math.round(nn * (x - xmin) / (xmax - xmin));
            xi[i] = Math.max(0, Math.min(nn - 1, xr));

            int yr = (int)Math.round(nn * (y - ymin) / (ymax - ymin));
            yi[i] = Math.max(0, Math.min(nn - 1, yr));
        }


        //    % localizations is discretized to computed the weights for resampling
        //                binned = accumarray(subs,1, bins);          % discretized image
        double[][] binned = new double[nn][nn];

        for (int i = 0; i < n; i++) {
            binned[xi[i]][yi[i]] += 1;
        }


        //    % smoothe the image
        //                f = fspecial('gaus',[round(7*fsize)+1 round(7*fsize)+1],fsize);
        //        localdensity = filter2(f,binned,'same');    % smoothed image
        int hsize = (int) Math.round(3.5 * fsize);
        double[][] localdensity;

        localdensity = gaussianFilter1DAndTranpose(binned, hsize, fsize);
        localdensity = gaussianFilter1DAndTranpose(localdensity, hsize, fsize);


        //    % weights for resampling function
        //        weights = zeros(1,S);
        //        for l = 1:S
        //        weights(1,l) = localdensity(xr(l),yr(l));
        //        end
        double[] weights = new double[n];
        for (int i = 0; i < n; i++) {
            weights[i] = localdensity[xi[i]][yi[i]];
        }


        //        % make sure that there is no NAN or negative weights
        //        weights(isnan(weights)) = 0;
        //        weights = max(weights,0);
        //        Max = numel(weights(weights > 0));
        for (int i = 0; i < n; i++) {
            if (Double.isNaN(weights[i]) || weights[i] < 0.0) {
                weights[i] = 0.0;
            }
        }


        //    % perform the weighted resampling
        //        ids = datasample(1:S,min(Max,cutoff),'Replace',false,'Weights',weights);
        //
        //    % new particle
        //        NewParticle.points = Particle.points(ids,:);
        //        NewParticle.sigma = Particle.sigma(ids,:);
        Integer[] indices = new Integer[n];
        Random random = new Random();
        for (int i = 0; i < n; i++) {
            indices[i] = i;
            weights[i] *= random.nextDouble();
        }

        Arrays.sort(indices, Comparator.comparingDouble(i -> -weights[i]));

        for (int i = 0; i < cutoff; i++) {
            int j = indices[i];

            resampledPos[2 * i + 0] = pos[2 * j + 0];
            resampledPos[2 * i + 1] = pos[2 * j + 1];
            resampledSigma[i] = sigmas[j];
        }
    }

    static private double[][] gaussianFilter1DAndTranpose(
            double[][] input,
            int radius,
            double sigma
    ) {
        double[] kernel = new double[radius + 1];
        double[][] output = new double[input.length][input[0].length];

        for (int i = 0; i < radius + 1; i++) {
            kernel[i] = Math.exp(-(i * i) / (2.0 * sigma * sigma));
        }

        for (int i = 0; i < input.length; i++) {
            int n = input[i].length;

            for (int j = 0; j < n; j++) {

                // Calculate Gaussian sum
                double sum = 0.0;
                for (int d = -radius; d <= radius; d++) {
                    if (j + d >= 0 && j + d < n) {
                        sum += input[i][j + d] * kernel[Math.abs(d)];
                    }
                }

                // And do tranpose
                output[j][i] = sum;
            }
        }

        return output;
    }

    public void cleanup() {
        //
    }
}

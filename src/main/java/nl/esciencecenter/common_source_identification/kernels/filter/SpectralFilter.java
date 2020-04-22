package nl.esciencecenter.common_source_identification.kernels.filter;

import edu.emory.mathcs.jtransforms.fft.FloatFFT_2D;
import jcuda.CudaException;
import jcuda.jcufft.JCufft;
import jcuda.jcufft.cufftHandle;
import jcuda.jcufft.cufftResult;
import jcuda.jcufft.cufftType;
import jcuda.runtime.cudaStream_t;
import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
import nl.esciencecenter.rocket.cubaapi.CudaStream;

public class SpectralFilter {

    protected CudaContext _context;
    protected CudaStream _stream;

    protected cufftHandle _planr2c;
    protected int _height;
    protected int _width;

    public SpectralFilter(int h, int w, CudaContext context, CudaStream stream) {

        _context = context;
        _stream = stream;
        _height = h;
        _width = w;
        _planr2c = new cufftHandle();

        //create CUFFT plan and associate withSupplier stream
        int res;
        res = JCufft.cufftPlan2d(_planr2c, h, w, cufftType.CUFFT_R2C);
        if (res != cufftResult.CUFFT_SUCCESS) {
            throw new CudaException(cufftResult.stringFor(res));
        }

        res = JCufft.cufftSetStream(_planr2c, new cudaStream_t(_stream.cuStream()));
        if (res != cufftResult.CUFFT_SUCCESS) {
            throw new CudaException(cufftResult.stringFor(res));
        }
    }

    public void applyGPU(CudaMemFloat input, CudaMemFloat d_output) {
        if (d_output.elementCount() < 2 * _height * (_width / 2 + 1)) {
            throw new IllegalArgumentException();
        }

        // fourier transform using CUFFT
        JCufft.cufftExecR2C(
                _planr2c,
                input.asDevicePointer(),
                d_output.asDevicePointer());
    }

    public void applyCPU(float[][] input, float[][] output) {
        float[][] tmp = new float[_height][2 * _width];

        for (int y = 0; y < _height; y++) {
            for (int x = 0; x < _width; x++) {
                tmp[y][2 * x + 0] = input[y][x];
                tmp[y][2 * x + 1] = 0;
            }
        }

        FloatFFT_2D bla = new FloatFFT_2D(_height, _width);
        bla.complexForward(tmp);

        for (int y = 0; y < _height; y++) {
            for (int x = 0; x < _width / 2 + 1; x++) {
                output[y][2 * x + 0] = tmp[y][2 * x + 0];
                output[y][2 * x + 1] = tmp[y][2 * x + 1];
            }
        }
    }

    public void cleanup() {
        JCufft.cufftDestroy(_planr2c);
    }

}

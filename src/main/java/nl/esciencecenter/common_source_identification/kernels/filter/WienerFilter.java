/*
* Copyright 2015 Netherlands eScience Center, VU University Amsterdam, and Netherlands Forensic Institute
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance withSupplier the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
package nl.esciencecenter.common_source_identification.kernels.filter;

import edu.emory.mathcs.jtransforms.fft.FloatFFT_1D;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_2D;
import nl.esciencecenter.rocket.util.Util;
import nl.esciencecenter.rocket.cubaapi.*;
import jcuda.*;
import jcuda.runtime.cudaStream_t;
import jcuda.jcufft.*;

import java.util.Arrays;

/**
 * Class for applying a series of Wiener Filters to a PRNU pattern 
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */
public class WienerFilter {

	protected CudaContext _context;
	protected CudaStream _stream;

	//handles to CUDA kernels
	protected CudaFunction _tocomplex;
	protected CudaFunction _toreal;
	protected CudaFunction _computeSquaredMagnitudes;
	protected CudaFunction _computeVarianceEstimates;
	protected CudaFunction _computeVarianceZeroMean;
	protected CudaFunction _sumFloats;
	protected CudaFunction _scaleWithVariances;
	protected CudaFunction _normalizeToReal;
	protected CudaFunction _normalizeComplex;

	//handle for CUFFT plan
	protected cufftHandle _planc2c;

	//handles to device memory arrays
	protected CudaMemFloat _d_comp;
	protected CudaMemFloat _d_sqmag;
	protected CudaMemFloat _d_varest;
	protected CudaMemFloat _d_variance;

	//parameterlists for kernel invocations
	protected Pointer toComplex;
	protected Pointer toReal;
	protected Pointer sqmag;
	protected Pointer varest;
	protected Pointer variancep;
	protected Pointer sumfloats;
	protected Pointer scale;
	protected Pointer normalize;
	protected Pointer normalizeComplex;

	protected int _height;
	protected int _width;
	
	/**
	 * Constructor for the Wiener Filter, used only by the PRNUFilter factory
	 * 
	 * @param h - the image height in pixels
	 * @param w - the image width in pixels
	 * @param context - the CudaContext as created by the factory
	 * @param stream - the CudaStream as created by the factory
	 * @param module - the CudaModule containing the kernels compiled by the factory
	 */
	public WienerFilter(int h, int w, CudaContext context, CudaStream stream, CudaModule module) {
		_context = context;
		_stream = stream;
		int n = h*w;
		this._height = h;
		this._width = w;

		//initialize CUFFT
		JCufft.initialize();
		JCufft.setExceptionsEnabled(true);
		_planc2c = new cufftHandle();

		//setup CUDA functions
		final int threads_x = 32;
		final int threads_y = 16;
		_tocomplex = module.getFunction("toComplex");
		_tocomplex.setDim(	(int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);
		
		_toreal = module.getFunction("toReal");
		_toreal.setDim(	(int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);
		
		_computeSquaredMagnitudes = module.getFunction("computeSquaredMagnitudes");
		_computeSquaredMagnitudes.setDim(	(int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);

		_computeVarianceEstimates = module.getFunction("computeVarianceEstimates");
		_computeVarianceEstimates.setDim(	(int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);
		
		final int threads = 128;
        final int nblocks = 1024;
		_computeVarianceZeroMean = module.getFunction("computeVarianceZeroMean");
		_computeVarianceZeroMean.setDim(	nblocks, 1, 1,
				threads, 1, 1);
		_sumFloats = module.getFunction("sumFloats");
		_sumFloats.setDim(	1, 1, 1,
				threads, 1, 1);

		_scaleWithVariances = module.getFunction("scaleWithVariances");
		_scaleWithVariances.setDim(	(int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);

		_normalizeToReal = module.getFunction("normalizeToReal");
		_normalizeToReal.setDim(	(int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);

		_normalizeComplex = module.getFunction("normalize");
		_normalizeComplex.setDim(	(int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);

		//allocate local variables in GPU memory
		_d_comp = _context.allocFloats(h*w*2);
		_d_sqmag = _context.allocFloats(h*w);
		_d_varest = _context.allocFloats(h*w);
		_d_variance = _context.allocFloats(nblocks);

		//create CUFFT plan and associate withSupplier stream
		int res;
		res = JCufft.cufftPlan2d(_planc2c, h, w, cufftType.CUFFT_C2C);
		if (res != cufftResult.CUFFT_SUCCESS) {
			throw new CudaException(cufftResult.stringFor(res));
		}
		res = JCufft.cufftSetStream(_planc2c, new cudaStream_t(_stream.cuStream()));
		if (res != cufftResult.CUFFT_SUCCESS) {
			throw new CudaException(cufftResult.stringFor(res));
		}

		sumfloats = Pointer.to(
				Pointer.to(_d_variance.asDevicePointer()),
				Pointer.to(_d_variance.asDevicePointer()),
				Pointer.to(new int[]{nblocks})
		);
		sqmag = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_sqmag.asDevicePointer()),
				Pointer.to(_d_comp.asDevicePointer())
		);
		varest = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_varest.asDevicePointer()),
				Pointer.to(_d_sqmag.asDevicePointer())
		);
		scale = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_comp.asDevicePointer()),
				Pointer.to(_d_comp.asDevicePointer()),
				Pointer.to(_d_varest.asDevicePointer()),
				Pointer.to(_d_variance.asDevicePointer())
		);
		normalizeComplex = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_comp.asDevicePointer()),
				Pointer.to(_d_comp.asDevicePointer())
		);
	}

	/**
	 * Applies the Wiener Filter to the input pattern already in GPU memory
	 */
	public void applyGPU(CudaMemFloat input) {
		int h = _height;
		int w = _width;

		//construct parameter lists for the CUDA kernels
		variancep = Pointer.to(
				Pointer.to(new int[]{h*w}),
				Pointer.to(_d_variance.asDevicePointer()),
				Pointer.to(input.asDevicePointer())
		);
		toReal = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(input.asDevicePointer()),
				Pointer.to(_d_comp.asDevicePointer())
		);
		normalize = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(input.asDevicePointer()),
				Pointer.to(_d_comp.asDevicePointer())
		);

		//convert values from real to complex
		_tocomplex.launch(_stream,
				h,
				w,
				_d_comp,
				input);

		//applyGPU complex to complex forward Fourier transform
		JCufft.cufftExecC2C(_planc2c, _d_comp.asDevicePointer(), _d_comp.asDevicePointer(), JCufft.CUFFT_FORWARD);

		//square the complex frequency values and store as real values
		_computeSquaredMagnitudes.launch(_stream, sqmag);

		//estimate local variances for four filter sizes, store minimum
		_computeVarianceEstimates.launch(_stream, varest);

		//compute global variance
		_computeVarianceZeroMean.launch(_stream, variancep);
		_sumFloats.launch(_stream, sumfloats);

		//scale the frequencies using global and local variance
		_scaleWithVariances.launch(_stream, scale);

		//inverse fourier transform using CUFFT
		JCufft.cufftExecC2C(_planc2c, _d_comp.asDevicePointer(), _d_comp.asDevicePointer(), JCufft.CUFFT_INVERSE);

		//CUFFT does not normalize the values after inverse transform, as such all values are scaled with N=(_height*_width)
		//normalize the values and convert from complex to real
		_normalizeToReal.launch(_stream, normalize);

	}

	/**
	 * Cleans up GPU memory and destroys FFT plan
	 */
	public void cleanup() {
		_d_comp.free();
		_d_sqmag.free();
		_d_varest.free();
		_d_variance.free();
		JCufft.cufftDestroy(_planc2c);
	}

	private static final int[] FILTER_SIZES = {3, 5, 7, 9};
	private static final int BORDER_SIZE = max(FILTER_SIZES) >> 1;

	public void applyCPU(final float[][] pixels) {
		float[][] varianceEstimates = pixels;
		int paddedWidth = (_width + (2 * BORDER_SIZE));
		FloatFFT_1D fftColumnTransform = new FloatFFT_1D(_height);
		FloatFFT_1D fftRowTransform = new FloatFFT_1D(_width);
		float[][] fft = new float[_height][_width * 2];
		float[] fftColumnBuffer = new float[2 * _height];
		float[][] squaredMagnitudes = new float[_height + (2 * BORDER_SIZE) + 1][paddedWidth];
		float[] sumSquareBuffer = new float[paddedWidth + 1];
		double sumSquares = 0.0;
		float[][] pixelsTransposed = Util.transpose(pixels);


		float[][] pixelsComplexTransposed = new float[_width][_height * 2];
		for (int i = 0; i < _width; i++) {
			for (int j = 0; j < _height; j++) {
				pixelsComplexTransposed[i][2*j] = pixelsTransposed[i][j];
				pixelsComplexTransposed[i][2*j+1] = 0.0f;
			}
		}

		FloatFFT_2D bla = new FloatFFT_2D(_width, _height);
		bla.complexForward(pixelsComplexTransposed);

		// Compute variance of input, perform FFT and initialize variance estimates
		// Note #1: We assume that the mean of the input data is zero (after 'zero mean').
		// Note #2: Both variance and variance estimates are scaled by 'n' to avoid divisions.
		for (int x = 0; x < _width; x++) {
			sumSquares += realColumnToComplex(pixels, x, fftColumnBuffer);

			fftColumnTransform.complexForward(fftColumnBuffer);
			storeComplexColumn(fftColumnBuffer, x, fft);
		}

		for (int y = 0; y < _height; y++) {
			fftRowTransform.complexForward(fft[y]);
		}

		float[][] pixelsComplex = Util.transposeComplex(pixelsComplexTransposed);

		for (int y = 0; y < _height; y++) {
			Arrays.fill(varianceEstimates[y], Float.MAX_VALUE);

			computeComplexMagnitudes(fft[y], squaredMagnitudes[y + BORDER_SIZE]);
		}

		// Estimate the minimum variance for each filter at each position
		for (final int filterSize : FILTER_SIZES) {
			updateVarianceEstimates(varianceEstimates, filterSize, sumSquareBuffer, squaredMagnitudes);
		}

		// 'Clean' the input using the minimum variance estimates and perform IFFT
		final int n = _width * _height;
		final float variance = (float) ((sumSquares * n) / (n - 1));
		for (int x = 0; x < _width; x++) {
			cleanColumn(varianceEstimates, variance, x, fftColumnBuffer, fft);

			fftColumnTransform.complexInverse(fftColumnBuffer, true);

			storeComplexColumn(fftColumnBuffer, x, fft);
		}
		for (int y = 0; y < _height; y++) {
			fftRowTransform.complexInverse(fft[y], true);

			complexToReal(fft[y], pixels[y], _width);
		}
	}

	private double realColumnToComplex(final float[][] pixels, final int x, final float[] column) {
		double sumSquares = 0.0;
		for (int y = 0; y < _height; y++) {
			final int idx2 = 2 * y;
			final float f = pixels[y][x];
			column[idx2] = f; // re
			column[idx2 + 1] = 0.0f; // im
			sumSquares += (f * f);
		}
		return sumSquares;
	}

	private void computeComplexMagnitudes(final float[] src, final float[] dest) {
		for (int x = 0; x < _width; x++) {
			final float re = src[x + x];
			final float im = src[x + x + 1];
			dest[BORDER_SIZE + x] = (re * re) + (im * im);
		}
	}

	private void cleanColumn(final float[][] varianceEstimates, final float variance, final int x, final float[] dest,
							 final float[][] src) {
		final int idx1 = 2 * x;
		for (int y = 0; y < _height; y++) {
			// Note: 'magScale' are the elements of 'Fmag1./Fmag' in the Matlab source!
			final float magScale = variance / Math.max(variance, varianceEstimates[y][x]);
			final int idx2 = 2 * y;
			dest[idx2] = src[y][idx1] * magScale;
			dest[idx2 + 1] = src[y][idx1 + 1] * magScale;
		}
	}

	private void storeComplexColumn(final float[] src, final int x, final float[][] dst) {
		final int idx1 = 2 * x;
		for (int y = 0; y < _height; y++) {
			final int idx2 = 2 * y;
			dst[y][idx1] = src[idx2];
			dst[y][idx1 + 1] = src[idx2 + 1];
		}
	}

	private static void complexToReal(final float[] complexSrc, final float[] realDest, final int size) {
		for (int i = 0; i < size; i++) {
			realDest[i] = complexSrc[i + i];
		}
	}

	private static int max(final int[] values) {
		int maxValue = values[0];
		for (int i = 1; i < values.length; i++) {
			maxValue = Math.max(maxValue, values[i]);
		}
		return maxValue;
	}

	private void updateVarianceEstimates(final float[][] varianceEstimates, final int filterSize,
										 final float[] _sumSquareBuffer, final float[][] _squaredMagnitudes) {
		int paddedWidth = (_width + (2 * BORDER_SIZE));
		final int borderOffset = BORDER_SIZE - ((filterSize - 1) / 2);
		final int paddedColumns = paddedWidth - (2 * borderOffset);
		final float fScale = 1.0f / (filterSize * filterSize);

		Arrays.fill(_sumSquareBuffer, 0.0f);

		// Compute sum of squares for each column for row (0)
		for (int y = 0; y < filterSize; y++) {
			final float[] rowMag = _squaredMagnitudes[y + borderOffset];
			for (int x = 0; x < paddedColumns; x++) {
				_sumSquareBuffer[x + 1] += rowMag[borderOffset + x];
			}
		}
		for (int x = 1; x <= paddedColumns; x++) {
			_sumSquareBuffer[x] *= fScale;
		}

		// Process rows
		for (int y = 0; y < _height; y++) {
			// Update minimum variance at each column
			float sumSquare = 0.0f;
			for (int x = 1; x < filterSize; x++) {
				sumSquare += _sumSquareBuffer[x];
			}
			final float[] rowEstimates = varianceEstimates[y];
			for (int x = 0; x < _width; x++) {
				sumSquare += (_sumSquareBuffer[x + filterSize] - _sumSquareBuffer[x]);

				// Note: If we assume that the mean of the samples is '0', we can use the sum of squares
				//       as estimate for the variance.
				rowEstimates[x] = Math.min(rowEstimates[x], sumSquare);
			}

			// Update sum of squares at each column for next row
			final float[] topSqMag = _squaredMagnitudes[y + borderOffset];
			final float[] bottomSqMag = _squaredMagnitudes[y + borderOffset + filterSize];
			for (int x = 0; x < paddedColumns; x++) {
				_sumSquareBuffer[x + 1] += fScale * (bottomSqMag[borderOffset + x] - topSqMag[borderOffset + x]);
			}
		}
	}
}

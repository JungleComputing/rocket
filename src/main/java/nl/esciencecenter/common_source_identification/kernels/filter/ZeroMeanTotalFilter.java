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

import nl.esciencecenter.rocket.cubaapi.*;
import jcuda.*;

/**
 * Class that applies a Zero Mean Total filter for filtering PRNU patterns
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 *
 */
public class ZeroMeanTotalFilter {

	protected CudaContext _context;
	protected CudaStream _stream;

	//handles to CUDA kernels
	protected CudaFunction _computeMeanVertically;
	protected CudaFunction _computeMeanHorizontally;

	//parameterlists for kernel invocations
	protected Pointer computeMeanVertically;
	protected Pointer computeMeanHorizontally;
	
	protected int _height;
	protected int _width;

	/**
	 * Constructor for the Zero Mean Total Filter, used only by the PRNUFilter factory
	 * 
	 * @param h - the image height in pixels
	 * @param w - the image width in pixels
	 * @param context - the CudaContext as created by the factory
	 * @param stream - the CudaStream as created by the factory
	 * @param module - the CudaModule containing the kernels compiled by the factory
	 */
	public ZeroMeanTotalFilter (int h, int w, CudaContext context, CudaStream stream, CudaModule module) {
		_context = context;
		_stream = stream;
		_height = h;
		_width = w;

		// Setup cuda functions
		final int block_size_x = 32;
		final int block_size_y = 16;

		_computeMeanVertically = module.getFunction("computeMeanVertically");
		_computeMeanVertically.setDim((int)Math.ceil(w / (float)block_size_x), 1, 1,
				block_size_x, block_size_y, 1);

		_computeMeanHorizontally = module.getFunction("computeMeanHorizontally");
		_computeMeanHorizontally.setDim(1, (int)Math.ceil(h / (float)block_size_y), 1,
				block_size_x, block_size_y, 1);
	}

	/**
	 * Applies the Zero Mean Total filter on the GPU.
	 * The input image is already in GPU memory and the output is also left
	 * on the GPU.
	 */
	public void applyGPU(CudaMemFloat input) {
		int h = _height;
		int w = _width;

		// Setup the parameter lists for each kernel call
		computeMeanVertically = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(input.asDevicePointer())
		);

		computeMeanHorizontally = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(input.asDevicePointer())
		);

		//applyGPU zero mean filter vertically
		_computeMeanVertically.launch(_stream, computeMeanVertically);

		//applyGPU the horizontal filter again to the transposed values
		_computeMeanHorizontally.launch(_stream, computeMeanHorizontally);
	}

    /**
     * cleans up GPU memory
     */
	public void cleanup() {
	}


	public void applyCPU(final float[][] pixels) {
		for (int y = 0; y < _height; y++) {
			filterRow(pixels[y]);
		}
		for (int x = 0; x < _width; x++) {
			filterColumn(pixels, x);
		}
	}

	private void filterRow(final float[] rowData) {
		float sumEven = 0.0f;
		float sumOdd = 0.0f;
		for (int i = 0; i < (_width - 1); i += 2) {
			sumEven += rowData[i];
			sumOdd += rowData[i + 1];
		}
		if (!isDivisibleByTwo(_width)) {
			sumEven += rowData[_width - 1];
		}

		final float meanEven = sumEven / ((_width + 1) >> 1);
		final float meanOdd = sumOdd / (_width >> 1);
		for (int i = 0; i < (_width - 1); i += 2) {
			rowData[i] -= meanEven;
			rowData[i + 1] -= meanOdd;
		}
		if (!isDivisibleByTwo(_width)) {
			rowData[_width - 1] -= meanEven;
		}
	}

	private void filterColumn(final float[][] pixels, final int x) {
		float sumEven = 0.0f;
		float sumOdd = 0.0f;
		for (int i = 0; i < (_height - 1); i += 2) {
			sumEven += pixels[i][x];
			sumOdd += pixels[i + 1][x];
		}
		if (!isDivisibleByTwo(_height)) {
			sumEven += pixels[_height - 1][x];
		}

		final float meanEven = sumEven / ((_height + 1) >> 1);
		final float meanOdd = sumOdd / (_height >> 1);
		for (int i = 0; i < (_height - 1); i += 2) {
			pixels[i][x] -= meanEven;
			pixels[i + 1][x] -= meanOdd;
		}
		if (!isDivisibleByTwo(_height)) {
			pixels[_height - 1][x] -= meanEven;
		}
	}

	private static boolean isDivisibleByTwo(final int value) {
		return (value & 1) == 0;
	}
}

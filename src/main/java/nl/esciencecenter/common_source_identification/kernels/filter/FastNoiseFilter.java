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
 * FastNoiseFilter for extraction PRNU pattern from an image
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 *
 */
public final class FastNoiseFilter {

	private static final float EPS = 1.0f;

	protected CudaContext _context;
	protected CudaStream _stream;

	//handles to CUDA kernels
	protected CudaFunction _normalized_gradient;
	protected CudaFunction _gradient;

	//handles to device memory arrays
	protected CudaMemFloat _d_dxs;
	protected CudaMemFloat _d_dys;

	//threads
	protected int _threads_x = 32;
	protected int _threads_y = 16;
	protected int _threads_z = 1;

	//grid
	protected int _grid_x;
	protected int _grid_y;
	protected int _grid_z;

	//parameterlists for kernel invocations
	protected Pointer normalized_gradient;
	protected Pointer gradient;

	protected int _height;
	protected int _width;

	/**
	 * Constructor for the FastNoise Filter, used only by the PRNUFilter factory
	 * 
	 * @param h - the image height in pixels
	 * @param w - the image width in pixels
	 * @param context - the CudaContext as created by the factory
	 * @param stream - the CudaStream as created by the factory
	 * @param module - the CudaModule containing the kernels compiled by the factory
	 */
	public FastNoiseFilter (int h, int w, CudaContext context, CudaStream stream, CudaModule module) {
		_context = context;
		_stream = stream;
		this._height = h;
		this._width = w;
		
		//setup grid dimensions
		_grid_x = (int)Math.ceil((float)w / (float)_threads_x);
		_grid_y = (int)Math.ceil((float)h / (float)_threads_y);
		_grid_z = 1;

		//setup cuda functions
		_normalized_gradient = module.getFunction("normalized_gradient");
		_normalized_gradient.setDim(_grid_x, _grid_y, _grid_z, _threads_x, _threads_y, _threads_z);

		_gradient = module.getFunction("gradient");
		_gradient.setDim(_grid_x, _grid_y, _grid_z, _threads_x, _threads_y, _threads_z);

		// Allocate the CUDA buffers for this kernel
		_d_dxs = _context.allocFloats(w*h);
		_d_dys = _context.allocFloats(w*h);

	}

	/**
	 * This method applies the FastNoise Filter on the GPU.
	 * The input is already in GPU memory.
	 * The output PRNU Noise pattern is stored in place of the input.
	 */
	public void applyGPU(CudaMemFloat input) {

		// Setup the parameter lists for each kernel call
		normalized_gradient = Pointer.to(
				Pointer.to(new int[]{_height}),
				Pointer.to(new int[]{_width}),
				Pointer.to(_d_dxs.asDevicePointer()),
				Pointer.to(_d_dys.asDevicePointer()),
				Pointer.to(input.asDevicePointer())
		);

		gradient = Pointer.to(
				Pointer.to(new int[]{_height}),
				Pointer.to(new int[]{_width}),
				Pointer.to(input.asDevicePointer()),
				Pointer.to(_d_dxs.asDevicePointer()),
				Pointer.to(_d_dys.asDevicePointer())
		);

		_normalized_gradient.launch(_stream, normalized_gradient);
		_gradient.launch(_stream, gradient);

	}

	/**
	 * cleans up GPU memory
	 */
	public void cleanup() {
		_d_dxs.free();
		_d_dys.free();
	}

	public void applyCPU(final float[][] pixels) {
		float[][] dx = new float[_height][_width];
		float[][] dy = new float[_height][_width];

		for (int y = 0; y < _height; y++) {
			computeHorizontalGradient(pixels[y], dx[y]);
		}
		for (int x = 0; x < _width; x++) {
			computeVerticalGradient(pixels, x, dy);
		}
		for (int y = 0; y < _height; y++) {
			normalizeGradients(dx[y], dy[y]);
		}
		for (int y = 0; y < _height; y++) {
			storeHorizontalGradient(dx[y], pixels[y]);
		}
		for (int x = 0; x < _width; x++) {
			addVerticalGradient(pixels, x, dy);
		}
	}

	private void computeHorizontalGradient(final float[] src, final float[] dest) {
		// Take forward differences on first and last element
		dest[0] = (src[1] - src[0]);
		dest[_width - 1] = (src[_width - 1] - src[_width - 2]);

		// Take centered differences on interior points
		for (int i = 1; i < (_width - 1); i++) {
			dest[i] = 0.5f * (src[i + 1] - src[i - 1]);
		}
	}

	private void computeVerticalGradient(final float[][] pixels, final int x, final float[][] _dy) {
		// Take forward differences on first and last element
		_dy[0][x] = (pixels[1][x] - pixels[0][x]);
		_dy[_height - 1][x] = (pixels[_height - 1][x] - pixels[_height - 2][x]);

		// Take centered differences on interior points
		for (int i = 1; i < (_height - 1); i++) {
			_dy[i][x] = 0.5f * (pixels[i + 1][x] - pixels[i - 1][x]);
		}
	}

	private void normalizeGradients(final float[] rowDx, final float[] rowDy) {
		for (int i = 0; i < _width; i++) {
			final float dx = rowDx[i];
			final float dy = rowDy[i];
			final float norm = (float) Math.sqrt((dx * dx) + (dy * dy));
			final float scale = 1.0f / (EPS + norm);
			rowDx[i] = (dx * scale);
			rowDy[i] = (dy * scale);
		}
	}

	private void storeHorizontalGradient(final float[] src, final float[] dest) {
		// Take forward differences on first and last element
		dest[0] = (src[1] - src[0]);
		dest[_width - 1] = (src[_width - 1] - src[_width - 2]);

		// Take centered differences on interior points
		for (int i = 1; i < (_width - 1); i++) {
			dest[i] = 0.5f * (src[i + 1] - src[i - 1]);
		}
	}

	private void addVerticalGradient(final float[][] dest, final int x, final float[][] _dy) {
		// Take forward differences on first and last element
		dest[0][x] += (_dy[1][x] - _dy[0][x]);
		dest[_height - 1][x] += (_dy[_height - 1][x] - _dy[_height - 2][x]);

		// Take centered differences on interior points
		for (int i = 1; i < (_height - 1); i++) {
			dest[i][x] += 0.5f * (_dy[i + 1][x] - _dy[i - 1][x]);
		}
	}
}

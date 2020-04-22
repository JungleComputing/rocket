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
 * Grayscale Filter class that takes an image as a BufferedImage
 * and converts it into a grayscale image stored as a float array
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */
public class GrayscaleFilter {

	private int _height;
	private int _width;

	private CudaContext _context;
	protected CudaStream _stream;

	//handles to CUDA kernels
	private CudaFunction _grayscale;

	/**
	 * Constructor for the Grayscale Filter, used only by the PRNUFilter factory
	 * 
	 * @param h - the image height in pixels
	 * @param w - the image width in pixels
	 * @param context - the CudaContext as created by the factory
	 * @param stream - the CudaStream as created by the factory
	 * @param module - the CudaModule containing the kernels compiled by the factory
	 */
	public GrayscaleFilter (int h, int w, CudaContext context, CudaStream stream, CudaModule module) {
		_context = context;
		_stream = stream;
		_height = h;
		_width = w;

		//setup cuda function
		final int threads_x = 32;
		final int threads_y = 16;
		_grayscale = module.getFunction("grayscale");
		_grayscale.setDim(      (int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);

	}

	/**
	 * Convert the image into a grayscaled image stored as an 1D float array on the GPU.
	 * The output is left in GPU memory for further processing.
	 *
	 * The conversion used currently is 0.299 r + 0.587 g + 0.114 b
	 *
	 * @param image - a BufferedImage that needs to be converted into grayscale
	 */
	public void applyGPU(CudaMem image, CudaMemFloat output) {
		if (image.sizeInBytes() < 3 * _width * _height) {
			throw new IllegalArgumentException();
		}

		Pointer grayscale = Pointer.to(
				Pointer.to(new int[]{_height}),
				Pointer.to(new int[]{_width}),
				Pointer.to(output.asDevicePointer()),
				Pointer.to(image.asDevicePointer()));

		//call GPU kernel to convert the color values to grayscaled float values
		_grayscale.launch(_stream, grayscale);
	}

	// reference implementation
	public float[][] applyCPU(byte[][][] colors, float[][] pixels) {
		for (int i = 0; i < _height; i++) {
			for (int j = 0; j < _width; j++) {
				// switch them around, because the byte array is b g r
				float b = (float) (colors[i][j][0] & 0xff);
				float g = (float) (colors[i][j][1] & 0xff);
				float r = (float) (colors[i][j][2] & 0xff);

				pixels[i][j] = 0.299f * r + 0.587f * g + 0.114f * b;
			}
		}

		return pixels;
	}

	/**
	 * cleans up allocated GPU memory
	 */
	public void cleanup() {
		//
	}

}

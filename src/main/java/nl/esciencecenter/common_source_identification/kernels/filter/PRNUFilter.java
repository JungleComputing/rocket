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

import java.io.IOException;
import java.net.URL;

import nl.esciencecenter.rocket.cubaapi.CudaContext;
import nl.esciencecenter.rocket.cubaapi.CudaMem;
import nl.esciencecenter.rocket.cubaapi.CudaMemByte;
import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
import nl.esciencecenter.rocket.cubaapi.CudaModule;
import nl.esciencecenter.rocket.cubaapi.CudaStream;

import jcuda.*;

/**
 * PRNUFilter is created for a specific image size. The CUDA source files
 * have been compiled by PRNUFilterFactory. Therefore, this object should
 * only be created using the Factory.
 * 
 * This class is used to instantiate the individual filter objects,
 * allocate GPU memory, etc.
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */
public class PRNUFilter {
	private static final String[] filenames = {
			"grayscalefilter.cu",
			"fastnoisefilter.cu",
			"zeromeantotalfilter.cu",
			"wienerfilter.cu"
	};

	GrayscaleFilter grayscaleFilter;
	FastNoiseFilter fastNoiseFilter;
	ZeroMeanTotalFilter zeroMeanTotalFilter;
	WienerFilter wienerFilter;
	SpectralFilter spectralFilter;
	
	protected int h;
	protected int w;
	protected boolean applySpectralFilter;

	protected CudaMemFloat d_image;
	private CudaModule[] modules;
	protected CudaStream stream;
	
	/**
	 * This constructor creates a CUDA stream for this filter, and
	 * instantiates the individual filters.
	 * 
	 * @param height - the image height
	 * @param width - the image width
	 * @param context - CudaContext object as created by PRNUFilterFactory
	 */
	public PRNUFilter(int height, int width, CudaContext context, boolean applySpectralFilter, String... compileArgs) throws CudaException, IOException {
		this.h = height;
		this.w = width;
		this.applySpectralFilter = applySpectralFilter;

		modules = new CudaModule[filenames.length];

		for (int i = 0; i < filenames.length; i++) {
			URL url = PRNUFilter.class.getResource(filenames[i]);
			modules[i] = context.compileModule(url, compileArgs);
		}

		//setup GPU memory
		//note that the filters also allocate memory for local variables
		d_image = context.allocFloats(height * width);
		
		//setup stream
		stream = new CudaStream();
		
        //instantiate individual filters
		grayscaleFilter		= new GrayscaleFilter(height, width, context, stream, modules[0]);
		fastNoiseFilter		= new FastNoiseFilter(height, width, context, stream, modules[1]);
		zeroMeanTotalFilter = new ZeroMeanTotalFilter(height, width, context, stream, modules[2]);
		wienerFilter		= new WienerFilter(height, width, context, stream, modules[3]);
		spectralFilter      = new SpectralFilter(height, width, context, stream);
		
	}

	public long getInputSize() {
		return h * w * 3 * Sizeof.BYTE;
	}

	public long getOutputSize() {
		int n;
		if (applySpectralFilter) {
			n = 2 * h * (w / 2 + 1); // Half the image as complex numbers
		} else {
			n = h * w;               // Entire image as floats
		}

		return Sizeof.FLOAT * n;
	}

	public void applyGPU(CudaMem input, CudaMem output) {
		applyGPU(input.asBytes(), output.asFloats());
	}

	/**
	 * This method applies all individual filters in order.
	 *
	 * @param rgbImage
	 * @return - a 1D float array containing the PRNU pattern of the input image
	 */
	public void applyGPU(CudaMemByte rgbImage, CudaMemFloat output) {
		try {
			grayscaleFilter.applyGPU(rgbImage, d_image);
			fastNoiseFilter.applyGPU(d_image);
			zeroMeanTotalFilter.applyGPU(d_image);
			wienerFilter.applyGPU(d_image);
			if (applySpectralFilter) {
				spectralFilter.applyGPU(d_image, output);
			} else {
				d_image.copyToDeviceAsync(output, stream);
			}
		} finally {
			stream.synchronize();
		}
	}

	public void applyCPU(byte[][][] rgbImage, float[][] output) {
		float[][] tmp = new float[h][w];

		grayscaleFilter.applyCPU(rgbImage, tmp);
		fastNoiseFilter.applyCPU(tmp);
		zeroMeanTotalFilter.applyCPU(tmp);
		wienerFilter.applyCPU(tmp);
		if (applySpectralFilter) {
			spectralFilter.applyCPU(tmp, output);
		} else {
			System.arraycopy(tmp, 0, output, 0, tmp.length);
		}
	}

	/*
	 * Getters for the individual filters
	 */
	public GrayscaleFilter getGrayscaleFilter() {
		return grayscaleFilter;
	}

	public FastNoiseFilter getFastNoiseFilter() {
		return fastNoiseFilter;
	}

	public ZeroMeanTotalFilter getZeroMeanTotalFilter() {
		return zeroMeanTotalFilter;
	}

	public WienerFilter getWienerFilter() {
		return wienerFilter;
	}

	public SpectralFilter getSpectralFilter() {
		return spectralFilter;
	}

	/**
	 * cleans up allocated GPU memory and other resources
	 */
	public void cleanup() {
		//call clean up methods of the filters
		grayscaleFilter.cleanup();
		fastNoiseFilter.cleanup();
		zeroMeanTotalFilter.cleanup();
		wienerFilter.cleanup();
		spectralFilter.cleanup();
		
		//free GPU memory
		d_image.free();
		
		//destroy stream
		stream.destroy();
	}
}

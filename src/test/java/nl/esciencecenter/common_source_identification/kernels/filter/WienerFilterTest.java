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

import nl.esciencecenter.rocket.util.Util;

import nl.esciencecenter.rocket.cubaapi.CudaMemFloat;
import org.junit.Test;

/**
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */
public class WienerFilterTest extends AbstractFilterTest {

	@Test
	public void applyGPUTest() {
		float[][] pixelsCPU = Util.copy(pixels);
		filter.getWienerFilter().applyCPU(pixelsCPU);

		float[] pixelsGPU = Util.from2DTo1D(HEIGHT, WIDTH, pixels);
		CudaMemFloat dmem = context.allocFloats(HEIGHT * WIDTH);

		dmem.copyFromHost(pixelsGPU);
		filter.getWienerFilter().applyGPU(dmem);
		dmem.copyToHost(pixelsGPU);

		dmem.free();

		//applyGPU CPU and GPU result
		assertArrayEquals(
				Util.from2DTo1D(HEIGHT, WIDTH, pixelsCPU),
				pixelsGPU);
	}
}

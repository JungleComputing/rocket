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
package nl.esciencecenter.common_source_identification.kernels.compare;

import jcuda.Pointer;
import nl.esciencecenter.rocket.cubaapi.CudaMem;

/**
 * This interface is used by objects that compute similarity scores of PRNU Patterns
 *
 * This interface was created in order to clean up the code in 
 * the main program. The NormalizedCrossCorrelation and 
 * PeakToCorrelationEnergy classes are used in the same way. 
 * With this interface all the code concerning the use of 
 * block-tiled loops can be reused for both.
 *
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 */


public interface PatternComparator {


   /***
     * This method performs an array of comparisons between patterns
     * It computes the PCE scores between all patterns in xPatterns and those in yPatterns
     *
     * @param left     PRNU pattern stored as float arrays
     * @param right    PRNU pattern stored as float arrays
     * @param result   the resulting score
     */
   public void applyGPU(CudaMem left, CudaMem right, CudaMem result);


    /***
     * Destroy this object. Delete memory, free resources, etc.
     */
   public void cleanup();
    

}

package nl.esciencecenter.rocket.util;

import ibis.constellation.ConstellationIdentifier;
import nl.esciencecenter.rocket.cubaapi.CudaContext;

import java.io.Serializable;

public class NodeInfo implements Serializable {
    static public class DeviceInfo implements Serializable {
        public String name;
        public long memorySize;
        public double loadingTime;
        public double parsingTime;
        public double preprocessingTime;
        public double execTime;

        public DeviceInfo(CudaContext context) {
            name = context.getDevice().getName();
            memorySize = context.getTotalMemory();
        }
    }

    public String name;
    public ConstellationIdentifier constellationIdentifier;
    public DeviceInfo[] devices;
}

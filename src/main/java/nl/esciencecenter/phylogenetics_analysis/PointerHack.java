package nl.esciencecenter.phylogenetics_analysis;

import com.sun.jna.Pointer;
import jcuda.NativePointerObject;
import nl.esciencecenter.rocket.cubaapi.CudaMem;
import nl.esciencecenter.rocket.cubaapi.CudaStream;

public class PointerHack {
    static class NativePointerProxy extends NativePointerObject {
        NativePointerProxy(NativePointerObject p) {
            super(p);
        }

        long nativePointer() {
            return getNativePointer();
        }
    }

    static class JCUDAPointerProxy extends jcuda.Pointer {
        JCUDAPointerProxy(jcuda.Pointer p) {
            super(p);
        }

        long byteOffset() {
            return getByteOffset();
        }
    }

    static public Pointer ptrOf(CudaMem mem) {
        return ptrOf(mem.asDevicePointer());
    }
    static public Pointer ptrOf(CudaStream stream) {
        return ptrOf(stream.cuStream());
    }

    static public Pointer ptrOf(NativePointerObject obj) {
        long ptr = new NativePointerProxy(obj).nativePointer();

        if (obj instanceof jcuda.Pointer) {
            ptr += new JCUDAPointerProxy((jcuda.Pointer) obj).byteOffset();
        }

        return new Pointer(ptr);
    }
}

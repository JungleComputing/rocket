package nl.esciencecenter.microscopy_particle_registration.util;

import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.FunctionVec;

import java.util.concurrent.ExecutorService;

abstract public class FunctionVecBase implements FunctionVec {

    @Override
    final public Vec f(double... x) {
        return f(new DenseVector(x));
    }

    @Override
    final public Vec f(Vec x, Vec s) {
        Vec p = f(x);
        p.copyTo(s);
        return s;
    }

    @Override
    final public Vec f(Vec x, Vec s, ExecutorService ex) {
        return f(x, s);
    }
}

/*
 * Copyright (c) 2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.esciencecenter.rocket.cubaapi;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import java.nio.Buffer;

import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.driver.CUfunction;

public final class CudaFunction {
	private final CUfunction _function;

	int _grid_x;
	int _grid_y;
	int _grid_z;
	int _threads_x;
	int _threads_y;
	int _threads_z;
	int _smem;

	CudaFunction(final CUfunction function) {
		_function = function;
	}

	public void setDim(int gx, int tx) {
		setDim(gx, 1, tx, 1);
	}

	public void setDim(int gx, int gy, int tx, int ty) {
		setDim(gx, gy, 1, tx, ty, 1);
	}

	public void setDim(int gx, int gy, int gz, int tx, int ty, int tz) {
		_grid_x = gx;
		_grid_y = gy;
		_grid_z = gz;
		_threads_x = tx;
		_threads_y = ty;
		_threads_z = tz;
	}

	public void setSharedMemory(int size) {
		_smem = size;
	}

	public void launch(CudaStream stream, Pointer parameters) {
		cuLaunchKernel(
				_function,
				_grid_x, _grid_y, _grid_z,
				_threads_x, _threads_y, _threads_z,
				_smem,
				stream.cuStream(),
				parameters,
				null);
	}

	public void launch(CudaStream stream, Object ...args) {
		NativePointerObject[] pargs = new Pointer[args.length];

		for (int i = 0; i < args.length; i++) {
			Object arg = args[i];
			NativePointerObject p;

			if (arg instanceof NativePointerObject) {
				p = (NativePointerObject) arg;
			} else if (arg instanceof Buffer) {
				p = Pointer.to((Buffer) arg);
			} else if (arg instanceof CudaMem) {
				p = Pointer.to(((CudaMem)arg).asDevicePointer());
			} else if (arg instanceof Byte) {
				p = Pointer.to(new byte[]{(Byte) arg});
			} else if (arg instanceof Short) {
				p = Pointer.to(new short[]{(Short) arg});
			} else if (arg instanceof Integer) {
				p = Pointer.to(new int[]{(Integer) arg});
			} else if (arg instanceof Long) {
				p = Pointer.to(new long[]{(Long) arg});
			} else if (arg instanceof Float) {
				p = Pointer.to(new float[]{(Float) arg});
			} else if (arg instanceof Double) {
				p = Pointer.to(new double[]{(Double) arg});
			} else if (arg == null) {
				throw new NullPointerException();
			} else {
				throw new IllegalArgumentException("invalid CudaFunction launch argument: " +
						arg.getClass());
			}

			pargs[i] = p;
		}

		launch(stream, Pointer.to(pargs));
	}

	public CUfunction CUfunction() {
		return _function;
	}

	@Override
	public String toString() {
		return _function.toString();
	}
}

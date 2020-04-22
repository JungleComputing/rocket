/*
 * Copyright (c) 2013-2014, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.esciencecenter.rocket.cubaapi;

import static jcuda.driver.JCudaDriver.cuModuleGetFunction;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;

import jcuda.CudaException;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

public final class CudaModule {
    private final CUmodule _module;
    private final byte[] _cubinData;

    CudaModule(final String cuSource, final String[] nvccOptions) {
        _module = new CUmodule();

        try {
            _cubinData = compileCuSourceToCubin(cuSource, nvccOptions);
        }
        catch (final IOException e) {
            throw new CudaException("Failed to compile CUDA source", e);
        }

        JCudaDriver.cuModuleLoadData(_module, _cubinData);
    }

    private static byte[] compileCuSourceToCubin(final String source, final String... options) throws IOException {
        final File cuFile = File.createTempFile("jcuda", ".cu");
        final File cubinFile = File.createTempFile("jcuda", ".cubin");

        try {
            FileUtils.write(cuFile, source);

            final List<String> arguments = new ArrayList<String>();
            arguments.add("nvcc");
            arguments.addAll(Arrays.asList(options));
            arguments.add("-cubin");
            arguments.add(cuFile.getAbsolutePath());
            arguments.add("-o");
            arguments.add(cubinFile.getAbsolutePath());

            final String output = runExternalCommand(arguments.toArray(new String[0]));
            if (output.length() > 2) {
                System.out.println(output);
            }

            //final String disassembly = runExternalCommand("cuobjdump", "-sass", cubinFile.getAbsolutePath());
            //System.out.println(disassembly);

            return FileUtils.readFileToByteArray(cubinFile);
        }
        finally {
            cuFile.delete();
            cubinFile.delete();
        }
    }

    private static String runExternalCommand(final String... arguments) throws IOException {
        final Process process = new ProcessBuilder().command(arguments).redirectErrorStream(true).start();
        final String processOutput = new String(IOUtils.toByteArray(process.getInputStream()));
        try {
            if (process.waitFor() != 0) {
                throw new IOException("Could not generate output file: " + processOutput);
            }
        }
        catch (final InterruptedException e) {
            throw new IOException("Interrupted while waiting for external process", e);
        }

        return processOutput;
    }

    public CudaFunction getFunction(final String name) {
        final CUfunction function = new CUfunction();
        cuModuleGetFunction(function, _module, name);
        return new CudaFunction(function);
    }
    
    public void cleanup() {
    	JCudaDriver.cuModuleUnload(_module);
    }

    @Override
    public String toString() {
        return _module.toString();
    }
}

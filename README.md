# Rocket: a Runtime System for All-Pair Computation on Heterogeneous Distributed Platforms
[![DOI](https://zenodo.org/badge/257305994.svg)](https://zenodo.org/badge/latestdoi/257305994)

This repository contains the code for _Rocket_. Rocket is a runtime system based on Constellation for efficient execution of all-pair computations on heterogeneous distributed platform. 

All-pair compute problems are a class of problems where one wants to compare each items from a data set to each other item in the data set using some user-defined comparison function. For example, given set of images, one could find all pairs of matching photos.

Rocket targets heterogeneous distributed platforms. In practice, this means any platform consisting of multiple nodes each equipped with at least one CUDA-enabled GPU.

Three example applications are included with Rocket from different scientific domains: a bioinformatics application, a digital forensics application, and localization microscopy application.


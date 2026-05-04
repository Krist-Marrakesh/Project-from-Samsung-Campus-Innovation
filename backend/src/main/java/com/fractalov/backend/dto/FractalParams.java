package com.fractalov.backend.dto;

public sealed interface FractalParams
        permits MandelbrotParams, JuliaParams, BurningShipParams, MultibrotParams {
    FractalType fractalType();

    int maxIter();

    double escapeRadius();

    boolean smoothing();
}

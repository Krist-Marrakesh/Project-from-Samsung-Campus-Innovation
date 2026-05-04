package com.fractalov.backend.service.render;

import com.fractalov.backend.dto.BurningShipParams;
import com.fractalov.backend.service.render.kernel.BurningShipKernel;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;

class BurningShipKernelTest {

    @Test
    void deNotSupported() {
        BurningShipKernel k = new BurningShipKernel(new BurningShipParams(200, 2.0, false));
        assertFalse(k.supportsDistanceEstimate(),
                "Burning Ship is non-holomorphic — DE must be advertised as unsupported");
    }
}

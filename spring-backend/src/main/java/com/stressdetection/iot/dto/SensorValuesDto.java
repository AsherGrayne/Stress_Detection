package com.stressdetection.iot.dto;

public record SensorValuesDto(
        double x,
        double y,
        double z,
        double eda,
        double hr,
        double temp
) {}

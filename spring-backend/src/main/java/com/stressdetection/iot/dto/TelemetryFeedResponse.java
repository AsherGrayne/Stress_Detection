package com.stressdetection.iot.dto;

public record TelemetryFeedResponse(
        long sequenceIndex,
        SensorValuesDto reading,
        int predictedStressCategory,
        String predictedStressLabel
) {}

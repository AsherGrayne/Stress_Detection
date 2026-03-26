package com.stressdetection.iot.dto;

import java.util.List;
import java.util.Map;

public record SimulatedLatestResponse(
        String observedAt,
        long sequenceIndex,
        SensorValuesDto reading,
        int predictedStressCategory,
        String predictedStressLabel,
        Map<String, List<TimeSeriesPointDto>> series
) {}

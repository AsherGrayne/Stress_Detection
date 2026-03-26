package com.stressdetection.iot.service;

import java.time.Instant;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.springframework.stereotype.Component;

import com.stressdetection.iot.config.StressProperties;
import com.stressdetection.iot.dto.SensorValuesDto;
import com.stressdetection.iot.dto.SimulatedLatestResponse;
import com.stressdetection.iot.dto.TimeSeriesPointDto;

@Component
public class SimulatedTelemetryState {

    private static final String[] KEYS = {"X", "Y", "Z", "EDA", "HR", "TEMP"};

    private final int maxPoints;
    private final Map<String, Deque<TimeSeriesPointDto>> series = new LinkedHashMap<>();

    private volatile SimulatedLatestResponse latest;

    public SimulatedTelemetryState(StressProperties stressProperties) {
        this.maxPoints = Math.max(10, stressProperties.getChartHistorySize());
        for (String k : KEYS) {
            series.put(k, new ArrayDeque<>(maxPoints + 1));
        }
    }

    public synchronized void append(
            Instant observedAt,
            long sequenceIndex,
            SensorValuesDto reading,
            int predictedStressCategory,
            String predictedStressLabel
    ) {
        long t = observedAt.toEpochMilli();
        push("X", t, reading.x());
        push("Y", t, reading.y());
        push("Z", t, reading.z());
        push("EDA", t, reading.eda());
        push("HR", t, reading.hr());
        push("TEMP", t, reading.temp());

        Map<String, List<TimeSeriesPointDto>> copy = new LinkedHashMap<>();
        for (String k : KEYS) {
            copy.put(k, new ArrayList<>(series.get(k)));
        }

        latest = new SimulatedLatestResponse(
                observedAt.toString(),
                sequenceIndex,
                reading,
                predictedStressCategory,
                predictedStressLabel,
                Map.copyOf(copy)
        );
    }

    private void push(String key, long t, double v) {
        Deque<TimeSeriesPointDto> d = series.get(key);
        d.addLast(new TimeSeriesPointDto(t, v));
        while (d.size() > maxPoints) {
            d.removeFirst();
        }
    }

    public SimulatedLatestResponse getSnapshot() {
        return latest;
    }
}

package com.stressdetection.iot.service;

import java.time.Instant;

import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import com.stressdetection.iot.StressCategoryLabels;
import com.stressdetection.iot.dto.IndexedSensorRow;
import com.stressdetection.iot.dto.SensorValuesDto;

@Component
public class SimulatedTelemetryScheduler {

    private final CsvTelemetryFeedService csvTelemetryFeedService;
    private final InferenceClient inferenceClient;
    private final SimulatedTelemetryState simulatedTelemetryState;
    private final StressLogService stressLogService;

    public SimulatedTelemetryScheduler(
            CsvTelemetryFeedService csvTelemetryFeedService,
            InferenceClient inferenceClient,
            SimulatedTelemetryState simulatedTelemetryState,
            StressLogService stressLogService
    ) {
        this.csvTelemetryFeedService = csvTelemetryFeedService;
        this.inferenceClient = inferenceClient;
        this.simulatedTelemetryState = simulatedTelemetryState;
        this.stressLogService = stressLogService;
    }

    @Scheduled(initialDelayString = "0", fixedRateString = "${stress.simulated-interval-ms:5000}")
    public void tick() {
        IndexedSensorRow row = csvTelemetryFeedService.takeNextRow();
        SensorValuesDto reading = row.reading();
        int category = inferenceClient.predict(
                reading.x(),
                reading.y(),
                reading.z(),
                reading.eda(),
                reading.hr(),
                reading.temp()
        );
        String label = StressCategoryLabels.labelForCategory(category);
        Instant now = Instant.now();
        simulatedTelemetryState.append(now, row.sequenceIndex(), reading, category, label);
        if (category >= 1) {
            stressLogService.logStress(now, category, label, reading, "simulated");
        }
    }
}

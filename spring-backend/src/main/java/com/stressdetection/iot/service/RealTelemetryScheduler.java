package com.stressdetection.iot.service;

import java.time.Instant;

import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import com.stressdetection.iot.StressCategoryLabels;
import com.stressdetection.iot.dto.FirebaseDataDto;
import com.stressdetection.iot.dto.SensorValuesDto;

@Component
public class RealTelemetryScheduler {

    private final InferenceClient inferenceClient;
    private final RealTelemetryState realTelemetryState;
    private final StressLogService stressLogService;
    private final RestTemplate restTemplate;

    private long sequenceIndex = 0;
    
    private static final String FIREBASE_URL = "https://stress-detection-c2bec-default-rtdb.asia-southeast1.firebasedatabase.app/data.json";

    public RealTelemetryScheduler(
            InferenceClient inferenceClient,
            RealTelemetryState realTelemetryState,
            StressLogService stressLogService
    ) {
        this.inferenceClient = inferenceClient;
        this.realTelemetryState = realTelemetryState;
        this.stressLogService = stressLogService;
        this.restTemplate = new RestTemplate();
    }

    @Scheduled(initialDelayString = "0", fixedRateString = "${stress.simulated-interval-ms:5000}")
    public void tick() {
        try {
            FirebaseDataDto data = restTemplate.getForObject(FIREBASE_URL, FirebaseDataDto.class);
            if (data == null || data.getAccX() == null) {
                return;
            }

            double x = data.getAccX() != null ? data.getAccX() : 0.0;
            double y = data.getAccY() != null ? data.getAccY() : 0.0;
            double z = data.getAccZ() != null ? data.getAccZ() : 0.0;
            double eda = data.getGsr() != null ? data.getGsr() : 0.0;
            double hr = data.getPulse() != null ? data.getPulse() : 0.0;
            double temp = data.getTemperature() != null ? data.getTemperature() : 0.0;

            SensorValuesDto reading = new SensorValuesDto(x, y, z, eda, hr, temp);
            
            int category = inferenceClient.predict(x, y, z, eda, hr, temp);
            String label = StressCategoryLabels.labelForCategory(category);
            Instant now = Instant.now();
            
            sequenceIndex++;
            realTelemetryState.append(now, sequenceIndex, reading, category, label);
            
            if (category >= 1) {
                stressLogService.logStress(now, category, label, reading, "real");
            }
        } catch (Exception e) {
            // Silently ignore or log failures to fetch from Firebase
            e.printStackTrace();
        }
    }
}

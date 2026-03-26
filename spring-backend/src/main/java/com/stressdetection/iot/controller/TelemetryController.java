package com.stressdetection.iot.controller;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import com.stressdetection.iot.dto.SimulatedLatestResponse;
import com.stressdetection.iot.dto.StressLogDto;
import com.stressdetection.iot.dto.TelemetryFeedResponse;
import com.stressdetection.iot.service.CsvTelemetryFeedService;
import com.stressdetection.iot.service.SimulatedTelemetryState;
import com.stressdetection.iot.service.RealTelemetryState;
import com.stressdetection.iot.service.StressLogService;
import org.springframework.web.bind.annotation.RequestParam;
import java.util.List;

@RestController
@RequestMapping("/api/telemetry")
public class TelemetryController {

    private final CsvTelemetryFeedService csvTelemetryFeedService;
    private final SimulatedTelemetryState simulatedTelemetryState;
    private final RealTelemetryState realTelemetryState;
    private final StressLogService stressLogService;

    public TelemetryController(
            CsvTelemetryFeedService csvTelemetryFeedService,
            SimulatedTelemetryState simulatedTelemetryState,
            RealTelemetryState realTelemetryState,
            StressLogService stressLogService
    ) {
        this.csvTelemetryFeedService = csvTelemetryFeedService;
        this.simulatedTelemetryState = simulatedTelemetryState;
        this.realTelemetryState = realTelemetryState;
        this.stressLogService = stressLogService;
    }

    /**
     * Latest simulated sample produced by the server every {@code stress.simulated-interval-ms}, with chart series
     * (timestamp = epoch ms on X, sensor value on Y).
     */
    @GetMapping("/simulated/latest")
    public SimulatedLatestResponse simulatedLatest() {
        SimulatedLatestResponse snap = simulatedTelemetryState.getSnapshot();
        if (snap == null) {
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "Simulated feed not ready yet");
        }
        return snap;
    }

    /**
     * Latest real sample from Firebase.
     */
    @GetMapping("/real/latest")
    public SimulatedLatestResponse realLatest() {
        SimulatedLatestResponse snap = realTelemetryState.getSnapshot();
        if (snap == null) {
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "Real feed not ready yet");
        }
        return snap;
    }

    /**
     * Returns the next row from balanced_data.csv as simulated device telemetry and the predicted stress class (0–2).
     */
    @GetMapping("/next")
    public TelemetryFeedResponse next() {
        return csvTelemetryFeedService.nextReading();
    }

    @GetMapping("/simulated/history")
    public List<StressLogDto> simulatedHistory(@RequestParam(defaultValue = "50") int limit) {
        return stressLogService.getHistory("simulated", limit);
    }

    @GetMapping("/real/history")
    public List<StressLogDto> realHistory(@RequestParam(defaultValue = "50") int limit) {
        return stressLogService.getHistory("real", limit);
    }
}

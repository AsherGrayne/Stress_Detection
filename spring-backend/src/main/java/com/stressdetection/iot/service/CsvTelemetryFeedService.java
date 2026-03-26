package com.stressdetection.iot.service;

import java.io.IOException;
import java.io.Reader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.springframework.stereotype.Service;

import com.stressdetection.iot.StressCategoryLabels;
import com.stressdetection.iot.config.StressProperties;
import com.stressdetection.iot.dto.IndexedSensorRow;
import com.stressdetection.iot.dto.SensorValuesDto;
import com.stressdetection.iot.dto.TelemetryFeedResponse;

import jakarta.annotation.PostConstruct;

@Service
public class CsvTelemetryFeedService {

    private final StressProperties stressProperties;
    private final InferenceClient inferenceClient;

    private List<double[]> rows = List.of();
    private int position;

    public CsvTelemetryFeedService(StressProperties stressProperties, InferenceClient inferenceClient) {
        this.stressProperties = stressProperties;
        this.inferenceClient = inferenceClient;
    }

    @PostConstruct
    public void loadCsv() {
        Path path = Paths.get(stressProperties.getCsvPath()).toAbsolutePath().normalize();
        if (!Files.isRegularFile(path)) {
            throw new IllegalStateException(
                    "balanced_data.csv not found at: " + path
                            + " — set stress.csv-path (e.g. absolute path or ../balanced_data.csv when running from spring-backend/)."
            );
        }
        List<double[]> loaded = new ArrayList<>();
        try (Reader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            Iterable<CSVRecord> records = CSVFormat.DEFAULT.builder()
                    .setHeader()
                    .setSkipHeaderRecord(true)
                    .build()
                    .parse(reader);
            for (CSVRecord rec : records) {
                loaded.add(
                        new double[] {
                                Double.parseDouble(rec.get("X")),
                                Double.parseDouble(rec.get("Y")),
                                Double.parseDouble(rec.get("Z")),
                                Double.parseDouble(rec.get("EDA")),
                                Double.parseDouble(rec.get("HR")),
                                Double.parseDouble(rec.get("TEMP")),
                        }
                );
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to read CSV: " + path, e);
        }
        if (loaded.isEmpty()) {
            throw new IllegalStateException("CSV has no data rows: " + path);
        }
        this.rows = List.copyOf(loaded);
    }

    /**
     * Next CSV row (cycles) as sensor values — advances the shared cursor (also used by the 5s scheduler).
     */
    public synchronized IndexedSensorRow takeNextRow() {
        int idx = position;
        position = (position + 1) % rows.size();
        double[] r = rows.get(idx);
        SensorValuesDto reading = new SensorValuesDto(r[0], r[1], r[2], r[3], r[4], r[5]);
        return new IndexedSensorRow(idx, reading);
    }

    /**
     * On-demand sample (advances cursor); prefer {@code GET /api/telemetry/simulated/latest} for Flutter.
     */
    public synchronized TelemetryFeedResponse nextReading() {
        IndexedSensorRow row = takeNextRow();
        SensorValuesDto reading = row.reading();
        int category = inferenceClient.predict(
                reading.x(),
                reading.y(),
                reading.z(),
                reading.eda(),
                reading.hr(),
                reading.temp()
        );
        return new TelemetryFeedResponse(
                row.sequenceIndex(),
                reading,
                category,
                StressCategoryLabels.labelForCategory(category)
        );
    }
}

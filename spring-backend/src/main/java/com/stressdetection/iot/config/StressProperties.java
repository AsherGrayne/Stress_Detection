package com.stressdetection.iot.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "stress")
public class StressProperties {

    /**
     * Path to balanced_data.csv (absolute, or relative to JVM working directory).
     */
    private String csvPath = "../balanced_data.csv";

    /**
     * Base URL of the Python inference service (no trailing slash).
     */
    private String inferenceBaseUrl = "http://127.0.0.1:8081";

    /**
     * Simulated telemetry tick interval (ms).
     */
    private long simulatedIntervalMs = 5000;

    /**
     * Max points per sensor chart (rolling window).
     */
    private int chartHistorySize = 60;

    /**
     * MongoDB connection URI (optional). If blank, high-stress logging is skipped.
     * Example: mongodb+srv://user:pass@cluster.../Stress_Dtabase?retryWrites=true&amp;w=majority
     */
    private String mongodbUri = "";

    public String getCsvPath() {
        return csvPath;
    }

    public void setCsvPath(String csvPath) {
        this.csvPath = csvPath;
    }

    public String getInferenceBaseUrl() {
        return inferenceBaseUrl;
    }

    public void setInferenceBaseUrl(String inferenceBaseUrl) {
        this.inferenceBaseUrl = inferenceBaseUrl;
    }

    public long getSimulatedIntervalMs() {
        return simulatedIntervalMs;
    }

    public void setSimulatedIntervalMs(long simulatedIntervalMs) {
        this.simulatedIntervalMs = simulatedIntervalMs;
    }

    public int getChartHistorySize() {
        return chartHistorySize;
    }

    public void setChartHistorySize(int chartHistorySize) {
        this.chartHistorySize = chartHistorySize;
    }

    public String getMongodbUri() {
        return mongodbUri;
    }

    public void setMongodbUri(String mongodbUri) {
        this.mongodbUri = mongodbUri;
    }
}

package com.stressdetection.iot.dto;

import java.util.Date;

public class StressLogDto {
    private int stressCategory;
    private String stressLabel;
    private Date loggedAt;
    private SensorValuesDto reading;

    public StressLogDto() {}

    public StressLogDto(int stressCategory, String stressLabel, Date loggedAt, SensorValuesDto reading) {
        this.stressCategory = stressCategory;
        this.stressLabel = stressLabel;
        this.loggedAt = loggedAt;
        this.reading = reading;
    }

    public int getStressCategory() {
        return stressCategory;
    }

    public void setStressCategory(int stressCategory) {
        this.stressCategory = stressCategory;
    }

    public String getStressLabel() {
        return stressLabel;
    }

    public void setStressLabel(String stressLabel) {
        this.stressLabel = stressLabel;
    }

    public Date getLoggedAt() {
        return loggedAt;
    }

    public void setLoggedAt(Date loggedAt) {
        this.loggedAt = loggedAt;
    }

    public SensorValuesDto getReading() {
        return reading;
    }

    public void setReading(SensorValuesDto reading) {
        this.reading = reading;
    }
}

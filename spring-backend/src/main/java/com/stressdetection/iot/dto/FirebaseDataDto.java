package com.stressdetection.iot.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class FirebaseDataDto {
    private Double temperature;
    private Double pulse;
    private Double gsr;
    private Double accX;
    private Double accY;
    private Double accZ;

    public Double getTemperature() { return temperature; }
    public void setTemperature(Double temperature) { this.temperature = temperature; }

    public Double getPulse() { return pulse; }
    public void setPulse(Double pulse) { this.pulse = pulse; }

    public Double getGsr() { return gsr; }
    public void setGsr(Double gsr) { this.gsr = gsr; }

    public Double getAccX() { return accX; }
    public void setAccX(Double accX) { this.accX = accX; }

    public Double getAccY() { return accY; }
    public void setAccY(Double accY) { this.accY = accY; }

    public Double getAccZ() { return accZ; }
    public void setAccZ(Double accZ) { this.accZ = accZ; }
}

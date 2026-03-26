package com.stressdetection.iot.dto;

public record IndexedSensorRow(long sequenceIndex, SensorValuesDto reading) {}

package com.stressdetection.iot.dto;

/**
 * @param t Epoch milliseconds (for chart X-axis: time).
 * @param v Sensor value (chart Y-axis).
 */
public record TimeSeriesPointDto(long t, double v) {}

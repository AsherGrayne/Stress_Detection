package com.stressdetection.iot.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * JSON body for POST /predict on the Python inference service.
 */
public record PredictRequest(
        @JsonProperty("x") double x,
        @JsonProperty("y") double y,
        @JsonProperty("z") double z,
        @JsonProperty("eda") double eda,
        @JsonProperty("hr") double hr,
        @JsonProperty("temp") double temp
) {}

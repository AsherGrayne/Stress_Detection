package com.stressdetection.iot.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record PredictResponse(
        @JsonProperty("stressCategory") int stressCategory
) {}

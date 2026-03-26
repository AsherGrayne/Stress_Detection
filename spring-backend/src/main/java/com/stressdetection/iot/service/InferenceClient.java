package com.stressdetection.iot.service;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.server.ResponseStatusException;

import com.stressdetection.iot.config.StressProperties;
import com.stressdetection.iot.dto.PredictRequest;
import com.stressdetection.iot.dto.PredictResponse;

@Service
public class InferenceClient {

    private final RestTemplate restTemplate;
    private final StressProperties stressProperties;

    public InferenceClient(RestTemplate restTemplate, StressProperties stressProperties) {
        this.restTemplate = restTemplate;
        this.stressProperties = stressProperties;
    }

    public int predict(
            double x,
            double y,
            double z,
            double eda,
            double hr,
            double temp
    ) {
        String url = stressProperties.getInferenceBaseUrl().replaceAll("/+$", "") + "/predict";
        PredictRequest body = new PredictRequest(x, y, z, eda, hr, temp);
        try {
            PredictResponse response = restTemplate.postForObject(url, body, PredictResponse.class);
            if (response == null) {
                throw new ResponseStatusException(
                        HttpStatus.SERVICE_UNAVAILABLE,
                        "Inference service returned empty body"
                );
            }
            return response.stressCategory();
        } catch (RestClientException e) {
            throw new ResponseStatusException(
                    HttpStatus.SERVICE_UNAVAILABLE,
                    "Inference service unavailable. Start Python: uvicorn inference_service.main:app --host 127.0.0.1 --port 8081 (from project root).",
                    e
            );
        }
    }
}

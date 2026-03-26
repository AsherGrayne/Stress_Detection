package com.stressdetection.iot;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.scheduling.annotation.EnableScheduling;

import com.stressdetection.iot.config.StressProperties;

@SpringBootApplication
@EnableScheduling
@EnableConfigurationProperties(StressProperties.class)
public class IotStressApplication {

    public static void main(String[] args) {
        SpringApplication.run(IotStressApplication.class, args);
    }
}

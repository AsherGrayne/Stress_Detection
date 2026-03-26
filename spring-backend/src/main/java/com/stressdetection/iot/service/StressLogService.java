package com.stressdetection.iot.service;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;

import org.bson.Document;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.stressdetection.iot.config.StressProperties;
import com.stressdetection.iot.dto.SensorValuesDto;
import com.stressdetection.iot.dto.StressLogDto;

import jakarta.annotation.PreDestroy;

@Service
public class StressLogService {

    private static final Logger log = LoggerFactory.getLogger(StressLogService.class);
    private static final String DB_NAME = "Stress_Detection";
    private static final String COLLECTION_SIM = "stress_log_simulated";
    private static final String COLLECTION_REAL = "stress_log_real";

    private final MongoClient client;
    private final MongoCollection<Document> collectionSimulated;
    private final MongoCollection<Document> collectionReal;

    public StressLogService(StressProperties stressProperties) {
        String uri = stressProperties.getMongodbUri();
        if (uri == null || uri.isBlank()) {
            log.warn(
                    "MongoDB URI not configured (set stress.mongodb-uri or MONGODB_URI). "
                            + "High-stress events will not be persisted."
            );
            this.client = null;
            this.collectionSimulated = null;
            this.collectionReal = null;
            return;
        }
        this.client = MongoClients.create(uri);
        MongoDatabase db = client.getDatabase(DB_NAME);
        this.collectionSimulated = db.getCollection(COLLECTION_SIM);
        this.collectionReal = db.getCollection(COLLECTION_REAL);
        log.info("MongoDB stress logging enabled (database={})", DB_NAME);
    }

    /**
     * Persists one document when stress category is Medium (1) or High (2).
     */
    public void logStress(Instant when, int category, String label, SensorValuesDto reading, String source) {
        MongoCollection<Document> targetCollection = "real".equalsIgnoreCase(source) ? collectionReal : collectionSimulated;
        if (targetCollection == null) {
            return;
        }
        Document r = new Document()
                .append("x", reading.x())
                .append("y", reading.y())
                .append("z", reading.z())
                .append("eda", reading.eda())
                .append("hr", reading.hr())
                .append("temp", reading.temp());
        Document doc = new Document()
                .append("stressCategory", category)
                .append("stressLabel", label)
                .append("loggedAt", Date.from(when))
                .append("reading", r);
        try {
            targetCollection.insertOne(doc);
        } catch (Exception e) {
            log.error("MongoDB insert failed for source {}", source, e);
        }
    }

    public List<StressLogDto> getHistory(String source, int limit) {
        MongoCollection<Document> targetCollection = "real".equalsIgnoreCase(source) ? collectionReal : collectionSimulated;
        if (targetCollection == null) {
            return Collections.emptyList();
        }
        List<StressLogDto> results = new ArrayList<>();
        try {
            targetCollection.find()
                    .sort(new Document("loggedAt", -1))
                    .limit(limit)
                    .forEach(doc -> {
                        Document r = (Document) doc.get("reading");
                        SensorValuesDto reading = new SensorValuesDto(
                                r.getDouble("x"),
                                r.getDouble("y"),
                                r.getDouble("z"),
                                r.getDouble("eda"),
                                r.getDouble("hr"),
                                r.getDouble("temp")
                        );
                        results.add(new StressLogDto(
                                doc.getInteger("stressCategory"),
                                doc.getString("stressLabel"),
                                doc.getDate("loggedAt"),
                                reading
                        ));
                    });
        } catch (Exception e) {
            log.error("MongoDB find failed for source {}", source, e);
        }
        return results;
    }

    @PreDestroy
    public void shutdown() {
        if (client != null) {
            client.close();
        }
    }
}

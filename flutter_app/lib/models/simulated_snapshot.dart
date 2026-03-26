class TimePoint {
  const TimePoint({required this.t, required this.v});

  final int t;
  final double v;

  factory TimePoint.fromJson(Map<String, dynamic> json) {
    return TimePoint(
      t: (json['t'] as num).toInt(),
      v: (json['v'] as num).toDouble(),
    );
  }
}

class SensorReading {
  const SensorReading({
    required this.x,
    required this.y,
    required this.z,
    required this.eda,
    required this.hr,
    required this.temp,
  });

  final double x;
  final double y;
  final double z;
  final double eda;
  final double hr;
  final double temp;

  factory SensorReading.fromJson(Map<String, dynamic> json) {
    return SensorReading(
      x: (json['x'] as num).toDouble(),
      y: (json['y'] as num).toDouble(),
      z: (json['z'] as num).toDouble(),
      eda: (json['eda'] as num).toDouble(),
      hr: (json['hr'] as num).toDouble(),
      temp: (json['temp'] as num).toDouble(),
    );
  }
}

class SimulatedSnapshot {
  const SimulatedSnapshot({
    required this.observedAt,
    required this.sequenceIndex,
    required this.reading,
    required this.predictedStressCategory,
    required this.predictedStressLabel,
    required this.series,
  });

  final String observedAt;
  final int sequenceIndex;
  final SensorReading reading;
  final int predictedStressCategory;
  final String predictedStressLabel;
  final Map<String, List<TimePoint>> series;

  factory SimulatedSnapshot.fromJson(Map<String, dynamic> json) {
    final seriesRaw = json['series'] as Map<String, dynamic>;
    final series = <String, List<TimePoint>>{};
    for (final e in seriesRaw.entries) {
      final list = (e.value as List<dynamic>)
          .map((x) => TimePoint.fromJson(x as Map<String, dynamic>))
          .toList();
      series[e.key] = list;
    }
    return SimulatedSnapshot(
      observedAt: json['observedAt'] as String,
      sequenceIndex: (json['sequenceIndex'] as num).toInt(),
      reading: SensorReading.fromJson(json['reading'] as Map<String, dynamic>),
      predictedStressCategory:
          (json['predictedStressCategory'] as num).toInt(),
      predictedStressLabel: json['predictedStressLabel'] as String,
      series: series,
    );
  }
}

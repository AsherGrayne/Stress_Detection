import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../models/simulated_snapshot.dart';

class SensorChartCard extends StatelessWidget {
  const SensorChartCard({
    super.key,
    required this.title,
    required this.points,
    required this.color,
  });

  final String title;
  final List<TimePoint> points;
  final Color color;

  @override
  Widget build(BuildContext context) {
    if (points.isEmpty) {
      return Card(
        margin: const EdgeInsets.only(bottom: 12),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Text('$title — waiting for data…'),
        ),
      );
    }

    final spots = points
        .map((p) => FlSpot(p.t.toDouble(), p.v))
        .toList(growable: false);
    final ts = points.map((p) => p.t).toList();
    var minT = ts.reduce((a, b) => a < b ? a : b).toDouble();
    var maxT = ts.reduce((a, b) => a > b ? a : b).toDouble();
    if (maxT <= minT) {
      minT -= 5000;
      maxT += 5000;
    }
    final vs = points.map((p) => p.v).toList();
    var minV = vs.reduce((a, b) => a < b ? a : b);
    var maxV = vs.reduce((a, b) => a > b ? a : b);
    if (minV == maxV) {
      minV -= 1;
      maxV += 1;
    }
    final pad = (maxV - minV) * 0.1;

    final timeFmt = DateFormat.Hms();

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(12, 16, 12, 8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 4),
            Text(
              'X: time  ·  Y: $title',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: Theme.of(context).colorScheme.outline,
                  ),
            ),
            const SizedBox(height: 12),
            SizedBox(
              height: 180,
              child: LineChart(
                LineChartData(
                  minX: minT,
                  maxX: maxT,
                  minY: minV - pad,
                  maxY: maxV + pad,
                  gridData: const FlGridData(show: true, drawVerticalLine: false),
                  borderData: FlBorderData(show: true),
                  titlesData: FlTitlesData(
                    topTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false),
                    ),
                    rightTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false),
                    ),
                    leftTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 44,
                        getTitlesWidget: (value, meta) {
                          return Text(
                            value.toStringAsFixed(
                              value.abs() >= 100 ? 0 : (value.abs() >= 10 ? 1 : 2),
                            ),
                            style: const TextStyle(fontSize: 10),
                          );
                        },
                      ),
                    ),
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 28,
                        interval: ((maxT - minT) / 3).clamp(1, double.infinity),
                        getTitlesWidget: (value, meta) {
                          final dt =
                              DateTime.fromMillisecondsSinceEpoch(value.toInt());
                          return Padding(
                            padding: const EdgeInsets.only(top: 6),
                            child: Text(
                              timeFmt.format(dt),
                              style: const TextStyle(fontSize: 9),
                            ),
                          );
                        },
                      ),
                    ),
                  ),
                  lineBarsData: [
                    LineChartBarData(
                      spots: spots,
                      isCurved: false,
                      color: color,
                      barWidth: 2.2,
                      dotData: FlDotData(
                        show: true,
                        getDotPainter: (spot, percent, bar, index) {
                          return FlDotCirclePainter(
                            radius: 4.5,
                            color: color,
                            strokeWidth: 2,
                            strokeColor: Theme.of(context)
                                .colorScheme
                                .surfaceContainerHighest,
                          );
                        },
                      ),
                      belowBarData: BarAreaData(show: false),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

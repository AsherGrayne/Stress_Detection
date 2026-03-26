import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../config.dart';
import 'history_screen.dart';
import '../models/simulated_snapshot.dart';
import '../services/telemetry_api.dart';
import '../widgets/sensor_chart_card.dart';

class RealScreen extends StatefulWidget {
  const RealScreen({super.key});

  @override
  State<RealScreen> createState() => _RealScreenState();
}

class _RealScreenState extends State<RealScreen> {
  final TelemetryApi _api = TelemetryApi();
  Timer? _timer;
  SimulatedSnapshot? _snapshot;
  String? _error;
  bool _loading = true;

  static const _order = ['X', 'Y', 'Z', 'EDA', 'HR', 'TEMP'];
  static const _colors = [
    Color(0xFF5C6BC0),
    Color(0xFF26A69A),
    Color(0xFFFFA726),
    Color(0xFFAB47BC),
    Color(0xFFEF5350),
    Color(0xFF42A5F5),
  ];

  @override
  void initState() {
    super.initState();
    _pull();
    _timer = Timer.periodic(const Duration(seconds: 5), (_) => _pull());
  }

  Future<void> _pull() async {
    try {
      final s = await _api.fetchRealLatest();
      if (!mounted) return;
      setState(() {
        _snapshot = s;
        _error = null;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    _api.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Real device telemetry'),
        actions: [
          IconButton(
            icon: const Icon(Icons.history_outlined),
            tooltip: 'View History Logs',
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(
                  builder: (_) => const HistoryScreen(source: 'real'),
                ),
              );
            },
          ),
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () {
              setState(() => _loading = true);
              _pull();
            },
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _pull,
        child: ListView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.all(16),
          children: [
            Text(
              'API: ${ApiConfig.baseUrl}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 8),
            if (_loading && _snapshot == null)
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(48),
                  child: CircularProgressIndicator(),
                ),
              )
            else if (_error != null && _snapshot == null)
              Card(
                color: Theme.of(context).colorScheme.errorContainer,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Text(
                    'Could not load data.\n\n$_error\n\n'
                    'Start Spring Boot (port 8090) and Python inference (8081).\n'
                    '${kIsWeb ? "Web uses http://localhost:8090 — do not use 10.0.2.2 in a browser.\n" : ""}'
                    'On a physical phone, run with: '
                    '--dart-define=API_BASE=http://YOUR_PC_LAN_IP:8090',
                  ),
                ),
              )
            else ...[
              _PredictionCard(snapshot: _snapshot!),
              const SizedBox(height: 16),
              const Text(
                'Sensor trends (time → X axis, value → Y axis)',
                style: TextStyle(fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 12),
              for (var i = 0; i < _order.length; i++)
                SensorChartCard(
                  title: _order[i],
                  points: _snapshot!.series[_order[i]] ?? const [],
                  color: _colors[i % _colors.length],
                ),
              if (_error != null)
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Text(
                    'Last refresh error: $_error',
                    style: TextStyle(
                      color: Theme.of(context).colorScheme.error,
                      fontSize: 12,
                    ),
                  ),
                ),
            ],
          ],
        ),
      ),
    );
  }
}

class _PredictionCard extends StatelessWidget {
  const _PredictionCard({required this.snapshot});

  final SimulatedSnapshot snapshot;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final cat = snapshot.predictedStressCategory;
    final (Color bg, Color fg) = switch (cat) {
      0 => (scheme.primaryContainer, scheme.onPrimaryContainer),
      1 => (scheme.secondaryContainer, scheme.onSecondaryContainer),
      _ => (scheme.errorContainer, scheme.onErrorContainer),
    };

    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Prediction',
              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 16),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: bg,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Category $cat',
                    style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                          color: fg,
                          fontWeight: FontWeight.bold,
                        ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    snapshot.predictedStressLabel,
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          color: fg,
                        ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    '0 — No Stress  ·  1 — Mild Stress  ·  2 — High Stress',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: fg.withValues(alpha: 0.85),
                        ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),
            Text(
              'Sample time: ${snapshot.observedAt}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            Text(
              'Sequence #${snapshot.sequenceIndex}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}

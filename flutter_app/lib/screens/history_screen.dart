import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import '../config.dart';

class HistoryScreen extends StatefulWidget {
  final String source;

  const HistoryScreen({super.key, required this.source});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  bool _loading = true;
  String? _error;
  List<dynamic> _history = [];

  @override
  void initState() {
    super.initState();
    _fetchHistory();
  }

  Future<void> _fetchHistory() async {
    try {
      final uri = Uri.parse('${ApiConfig.baseUrl}/api/telemetry/${widget.source}/history');
      final res = await http.get(uri);
      
      if (res.statusCode != 200) {
        throw Exception('HTTP ${res.statusCode}: ${res.body}');
      }
      
      if (!mounted) return;
      
      setState(() {
        _history = jsonDecode(res.body) as List<dynamic>;
        _loading = false;
        _error = null;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = e.toString();
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final title = widget.source == 'real' ? 'Real History' : 'Simulated History';
    
    return Scaffold(
      appBar: AppBar(
        title: Text(title),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () {
              setState(() => _loading = true);
              _fetchHistory();
            },
          ),
        ],
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_loading && _history.isEmpty) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_error != null && _history.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Text('Failed to load history: $_error\nMake sure backend is running.'),
        ),
      );
    }
    
    if (_history.isEmpty) {
      return const Center(
        child: Text('No medium or high stress events logged yet.'),
      );
    }

    return ListView.separated(
      padding: const EdgeInsets.all(16),
      itemCount: _history.length,
      separatorBuilder: (_, __) => const SizedBox(height: 12),
      itemBuilder: (context, index) {
        final item = _history[index];
        final cat = item['stressCategory'];
        final label = item['stressLabel'];
        final date = item['loggedAt'];
        final reading = item['reading'] ?? {};
        
        final scheme = Theme.of(context).colorScheme;
        final (Color bg, Color fg) = cat == 1
            ? (scheme.secondaryContainer, scheme.onSecondaryContainer)
            : (scheme.errorContainer, scheme.onErrorContainer);

        return Card(
          color: bg,
          margin: EdgeInsets.zero,
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      label,
                      style: TextStyle(color: fg, fontWeight: FontWeight.bold, fontSize: 16),
                    ),
                    Text(
                      DateTime.tryParse(date ?? '')?.toLocal().toString().split('.')[0] ?? 'Unknown time',
                      style: TextStyle(color: fg.withValues(alpha: 0.8), fontSize: 13),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Text(
                  'EDA: ${reading["eda"]}  |  HR: ${reading["hr"]}  |  Temp: ${reading["temp"]}',
                  style: TextStyle(color: fg),
                ),
                Text(
                  'Acc (X,Y,Z): ${reading["x"]}, ${reading["y"]}, ${reading["z"]}',
                  style: TextStyle(color: fg),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

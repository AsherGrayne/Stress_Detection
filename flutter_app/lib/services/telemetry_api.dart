import 'dart:convert';

import 'package:http/http.dart' as http;

import '../config.dart';
import '../models/simulated_snapshot.dart';

class TelemetryApi {
  TelemetryApi({http.Client? client}) : _client = client ?? http.Client();

  final http.Client _client;

  Uri get _latestUri => Uri.parse('${ApiConfig.baseUrl}/api/telemetry/simulated/latest');
  Uri get _realLatestUri => Uri.parse('${ApiConfig.baseUrl}/api/telemetry/real/latest');

  Future<SimulatedSnapshot> fetchSimulatedLatest() async {
    final res = await _client.get(_latestUri);
    if (res.statusCode != 200) {
      throw TelemetryException(
        'HTTP ${res.statusCode}: ${res.body}',
        statusCode: res.statusCode,
      );
    }
    final map = jsonDecode(res.body) as Map<String, dynamic>;
    return SimulatedSnapshot.fromJson(map);
  }

  Future<SimulatedSnapshot> fetchRealLatest() async {
    final res = await _client.get(_realLatestUri);
    if (res.statusCode != 200) {
      throw TelemetryException(
        'HTTP ${res.statusCode}: ${res.body}',
        statusCode: res.statusCode,
      );
    }
    final map = jsonDecode(res.body) as Map<String, dynamic>;
    return SimulatedSnapshot.fromJson(map);
  }

  void close() => _client.close();
}

class TelemetryException implements Exception {
  TelemetryException(this.message, {this.statusCode});

  final String message;
  final int? statusCode;

  @override
  String toString() => message;
}

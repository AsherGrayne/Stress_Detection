import 'package:flutter/foundation.dart';

/// Spring Boot base URL (no trailing slash).
///
/// **Override** (always wins): `flutter run --dart-define=API_BASE=http://YOUR:8090`
///
/// Defaults:
/// - **Web** (Chrome/Edge) → `http://localhost:8090` (`10.0.2.2` does not work in a browser)
/// - **Android emulator** → `http://10.0.2.2:8090` (host machine)
/// - **iOS simulator / Windows / macOS / Linux** → `http://127.0.0.1:8090`
/// - **Physical device** → set `API_BASE` to your PC’s LAN IP, e.g. `http://192.168.1.5:8090`
class ApiConfig {
  ApiConfig._();

  static const String _fromEnv = String.fromEnvironment('API_BASE');

  static String get baseUrl {
    if (_fromEnv.isNotEmpty) {
      return _fromEnv;
    }
    if (kIsWeb) {
      return 'http://localhost:8090';
    }
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return 'http://10.0.2.2:8090';
      default:
        return 'http://127.0.0.1:8090';
    }
  }
}

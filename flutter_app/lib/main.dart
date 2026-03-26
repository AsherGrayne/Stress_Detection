import 'package:flutter/material.dart';

import 'screens/home_screen.dart';

void main() {
  runApp(const StressIotApp());
}

class StressIotApp extends StatelessWidget {
  const StressIotApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Stress IoT',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF1565C0)),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

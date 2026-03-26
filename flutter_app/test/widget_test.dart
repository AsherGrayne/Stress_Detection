import 'package:flutter_test/flutter_test.dart';
import 'package:stress_iot_app/main.dart';

void main() {
  testWidgets('App loads home with Simulated button', (WidgetTester tester) async {
    await tester.pumpWidget(const StressIotApp());
    expect(find.text('Simulated'), findsOneWidget);
    expect(find.text('Real'), findsOneWidget);
  });
}

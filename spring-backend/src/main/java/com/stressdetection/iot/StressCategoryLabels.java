package com.stressdetection.iot;

public final class StressCategoryLabels {

    private StressCategoryLabels() {}

    public static String labelForCategory(int category) {
        return switch (category) {
            case 0 -> "No Stress";
            case 1 -> "Mild Stress";
            case 2 -> "High Stress";
            default -> "Unknown (" + category + ")";
        };
    }
}

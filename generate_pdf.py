from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def create_transcript_pdf(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = styles['Title']
    title_style.fontSize = 18
    title_style.spaceAfter = 20

    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        textColor='#2E5090'
    )

    normal_style = styles['Normal']
    normal_style.fontSize = 11
    normal_style.leading = 14
    normal_style.spaceAfter = 12

    Story = []

    # Title
    Story.append(Paragraph("Presentation Transcript", title_style))
    Story.append(Paragraph("An Industrial IoT Monitoring System for Anomaly Detection Using Machine Learning Algorithms", styles['Heading3']))
    Story.append(Spacer(1, 0.2 * inch))

    # Data
    slides = [
        ("Slide 1: Title Slide",
         "Hello everyone, and welcome. My name is [Your Name], and today I am excited to present our research titled 'An Industrial IoT Monitoring System for Anomaly Detection Using Machine Learning Algorithms.' This work was conducted alongside my co-authors at the Department of Computer Science and Engineering, Assam Engineering College. Our research focuses on bridging the gap between raw industrial sensor data and real-time, intelligent predictive maintenance."),
        
        ("Slide 2: Introduction & Motivation",
         "To give you some background on why this matters: unexpected equipment failures can be incredibly costly, sometimes draining up to 15% of an industrial facility's annual revenue due to unplanned downtime. Historically, industries have relied on traditional SCADA architectures that use simple static threshold monitoring. For example, an alarm might only trigger when a temperature crosses a fixed limit. The problem with this approach is that it is rigid; it overlooks early, subtle degradation signals caused by equipment aging or seasonal shifts, leading to missed faults or false alarms. Our motivation was to transition from this reactive, calendar-driven maintenance to a proactive, condition-based predictive paradigm by bringing intelligence directly to the edge."),
        
        ("Slide 3: The Proposed Solution",
         "To address these limitations, we propose an integrated Industrial Internet of Things (IIoT) monitoring platform. Our solution seamlessly combines real-time multi-sensor data collection with a robust machine learning inference layer. Instead of just looking at raw, instantaneous values, our system applies a sliding window feature extraction technique to evaluate live readings across seven critical parameters. We run these features through five parallel machine learning baseline models simultaneously, all while strictly maintaining an end-to-end inference latency of under 200 milliseconds to support a live, operational web dashboard."),
        
        ("Slide 4: Hardware Setup & IoT Layer",
         "Let's dive into the foundational layer of our system: the hardware and IoT setup. We designed a practical, low-cost prototype to demonstrate continuous industrial monitoring. At the core, we use an Arduino Uno acting as the primary controller. It is interfaced with a DHT11 sensor to capture ambient temperature and humidity, and an ACS712 Hall-effect sensor to measure the electrical current drawn by the load. These sensors continuously sample the environment, package the readings as JSON payloads, and transmit them over USB serial communication at a 9600 baud rate to our host system."),
        
        ("Slide 5: High-Level System Architecture",
         "This brings us to our high-level architecture, which is divided into four main modules. First, the IoT Layer captures the raw physical signals. Second, the Data Processing Layer takes this serial stream, cleans it, adds timestamps, and segments the data into overlapping windows. Third, the Machine Learning Layer extracts statistical features from these windows and passes them to our trained models to detect abnormal operating patterns. Finally, the Web Application Layer translates these complex ML outputs into an intuitive, real-time browser dashboard for human operators."),
        
        ("Slide 6: Hardware and Data Acquisition Setup",
         "Looking closer at the data acquisition flow, the raw streams are heavily processed before any model sees them. Because individual data points in time-series data often lack context, we compute sliding-window statistics over 24-sample overlapping windows. By extracting statistical descriptors like mean, standard deviation, skewness, and kurtosis over these windows, we successfully preserve short-term temporal patterns while keeping the dimensionality manageable for our algorithms to process in real-time."),
        
        ("Slide 7: Methodology & Feature Engineering",
         "For our methodology, we evaluated our system using both synthetic and real-world strategies. Initially, we generated 30 days of simulated telemetry data at one-minute intervals, covering 43,200 values across seven signals like vibration, pressure, viscosity, and power. We applied a controlled 5% anomaly injection strategy to simulate practical fault signatures, such as Gaussian deviations and spike disturbances. With our window size set to 24 steps, we extracted 24 features per window, creating a structured data matrix ready for classification."),
        
        ("Slide 8: Evaluated Machine Learning Algorithms",
         "We evaluated five distinct machine learning algorithms to ensure a comprehensive comparison across different modeling paradigms rather than just hyperparameter tuning. We selected Isolation Forest as an unsupervised learner, ideal for cold-start scenarios where anomalous data isn't available. We also tested a kernel-based Support Vector Machine (SVM) to handle non-linear relations. Finally, we included ensemble methods like Gradient Boosting and Random Forest, alongside a linear baseline, Logistic Regression, to see how different architectures handle industrial time-series data."),
        
        ("Slide 9: Experimental Results (Synthetic Benchmark)",
         "Our initial experimental results on the synthetic benchmark were highly promising. Compared to traditional threshold-based systems that typically cap at 60% to 75% accuracy, our supervised models—SVM, Random Forest, and Logistic Regression—achieved an outstanding average accuracy of 99.35% across all seven sensor parameters. Gradient Boosting closely followed at 98.73%. Importantly, we also analyzed Precision, Recall, and F1-Scores to account for extreme dataset imbalance, and verified these results using 5-fold stratified cross-validation and ROC-AUC trends."),
        
        ("Slide 10: Real-World Dataset Validation",
         "However, to ensure practical reliability and address domain shift, we took our pipeline a step further and validated it against a real-world industrial water pump dataset containing over 44,000 telemetry samples across 50 sensor channels. Using a Remaining Useful Life (RUL) based binary labeling approach, we tested the unsupervised Isolation Forest model, which achieved a 66.34% accuracy. This real-world test confirmed our system's end-to-end operability on actual noisy data, while highlighting the inherent difficulties of detecting weakly labeled, multi-sensor correlated faults in the field."),
        
        ("Slide 11: Load Testing & Visualization Dashboard",
         "To prove the system's readiness for deployment, we conducted rigorous load testing. Under single-stream operation, the system effortlessly sustained 298.5 requests per second with a 99th-percentile latency of just 5.16 milliseconds and zero failures. All of this rapid inference feeds directly into our live web dashboard. The dashboard features interactive gauges, rolling time-series graphs, and an automated email alert mechanism. It allows operators to visually confirm ML predictions and instantly spot anomalies before equipment fails."),
        
        ("Slide 12: Conclusion & Future Scope",
         "In conclusion, we have successfully demonstrated an end-to-end, edge-to-dashboard IIoT predictive maintenance system. By combining low-cost sensors with optimized machine learning algorithms, we achieved sub-200 millisecond latencies and high detection accuracies. While our supervised models excel with synthetic data, our real-world validation shows that unsupervised methods like Isolation Forest are vital when labeled faults are scarce. For future scope, we aim to explore richer fault labels, adaptive feature extraction, multi-process or GPU-accelerated inference for higher concurrency, and eventually scale to cloud-assisted multi-site monitoring."),
        
        ("Slide 13 & 14: References",
         "These are the key references and literature that formed the foundation of our research. Thank you very much for your time and attention. I would now be happy to take any questions.")
    ]

    for title, content in slides:
        Story.append(Paragraph(title, heading_style))
        Story.append(Paragraph(content, normal_style))
        Story.append(Spacer(1, 0.1 * inch))

    # Presentation Tips
    Story.append(Spacer(1, 0.3 * inch))
    Story.append(Paragraph("Tips for your presentation:", styles['Heading3']))
    
    tips = [
        "<b>Pacing:</b> Practice speaking at a steady, deliberate pace. This transcript provides enough content for about a 10 to 15-minute presentation.",
        "<b>Engagement:</b> When discussing the Dashboard (Slide 11) or the Hardware setup (Slides 4 & 6), point to the specific images or diagrams on your slide to direct the audience's attention.",
        "<b>Tone:</b> The text is structured to sound academic yet accessible. Feel free to tweak minor transition words to better match your natural speaking style!"
    ]
    
    for tip in tips:
        Story.append(Paragraph(f"• {tip}", normal_style))

    doc.build(Story)

if __name__ == '__main__':
    create_transcript_pdf('Presentation_Transcript.pdf')

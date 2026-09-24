/*
Demo arduino code for wheel functionality
*/

// 3 input pins:
// dir: 1/0 for cloclwise/counterclockwise
// br: breaking while high
// PWM: motor strength
//      Max speed: 10,000 RPM
//      TODO: find max PWM value for max speed
// Others are motor settings that will be hardcoded later

// Output:
// Freq: hall sensor output--high every time wheel rotates
#define DAA 10
#define COMU 5
#define FREQ 2
#define PWM 3
#define BR 11
#define DIR 6

const int numberPolePairs = 4;
unsigned long lastReportTime = 0;
volatile unsigned long lastPulseTime = 0;
volatile unsigned long pulseInterval = 0;
volatile boolean newPulse = false;
volatile double rpm = 0.0;

void setup() {
    //Set up pins
    pinMode(DAA, OUTPUT);
    pinMode(COMU, OUTPUT);
    pinMode(FREQ, INPUT);
    pinMode(PWM, OUTPUT);
    pinMode(BR, OUTPUT);
    pinMode(DIR, OUTPUT);

    digitalWrite(BR, LOW);
    digitalWrite(DIR, LOW);
    analogWrite(PWM, 128);
    digitalWrite(DAA, LOW);
    digitalWrite(COMU, LOW);

    delay(10000);

    //Stuff
    Serial.begin(9600);
    attachInterrupt(digitalPinToInterrupt(FREQ), readFG, RISING);
}

void loop() {
    for (int pwm = 63; pwm <= 255; pwm += 64) {

        analogWrite(PWM, pwm);
        delay(4000);

        for (int daa = 0; daa <= 2; daa++) {
            pinMode(DAA, daa < 2 ? OUTPUT : INPUT);
        if (daa < 2) {
            digitalWrite(DAA, daa);
        } else {
            digitalWrite(DAA, LOW);
        }

        for (int comu = 0; comu <= 2; comu++) {

            pinMode(COMU, comu < 2 ? OUTPUT : INPUT);

            if (comu < 2) {
                digitalWrite(COMU, comu);
            } else {
                digitalWrite(COMU, LOW);
            }

            delay(2000);

            Serial.print("PWM: ");
            Serial.print(pwm);
            Serial.print(" DAA: ");
            Serial.print(daa);
            Serial.print(" COMU: ");
            Serial.print(comu);
            Serial.print(" RPM: ");
            Serial.println(rpm);
            }
        }
    }

    digitalWrite(BR, LOW);
    analogWrite(PWM, 255);
    delay(5000);

    digitalWrite(PWM, LOW);
    delay(10000);

    digitalWrite(BR, HIGH);
    analogWrite(PWM, 255);
    delay(5000);

    digitalWrite(PWM, LOW);
    delay(5000);
}

// Interrupt Service Routine (ISR)
void readFG() {
    unsigned long currentTime = micros();

    // Calculate time since last pulse
    pulseInterval = currentTime - lastPulseTime;
    lastPulseTime = currentTime;

    // Disable interrupts briefly to read volatile variables safely
    noInterrupts();
    unsigned long intervalCopy = pulseInterval;
    newPulse = false;

    interrupts();

    // Avoid division by zero
    if (intervalCopy > 0) {
        // 1. Calculate Frequency (Hz)
        // intervalCopy is in microseconds, so 1,000,000 / interval = Hz
        float frequencyFG = 1000000.0 / intervalCopy;

        // 2. Calculate RPM
        // Formula: (Freq * 60) / PolePairs
        rpm = (frequencyFG * 60.0) / numberPolePairs;
    }
}

#include <ServoTimer2.h>  // uses Timer2

// ===== USER CONFIGURABLE MOTOR PWM FREQUENCY =====
// 31.25 kHz → 511
// 25.00 kHz → 639
// 20.00 kHz → 799
// 10.00 kHz → 1599
//  5.00 kHz → 3199
//  1.00 kHz → 15999
#define MOTOR_PWM_TOP 1599   // <-- change this for frequency

// ===== USER CONFIGURABLE MOTOR SPEED BOUNDS =====
// Forward motion
#define MOTOR_FWD_MIN  0.09f
#define MOTOR_FWD_MAX  0.75f
// Backward motion
#define MOTOR_REV_MIN -0.10f
#define MOTOR_REV_MAX -0.25f

// ===== Headlight config =====
// Pin 6 uses Timer0 PWM at ~976 Hz (≈1 kHz) by default on Arduino Uno.
// This stays within your 1 kHz to 10 kHz requirement without touching other timers.
const int HEADLIGHT_PWM_PIN = 6;
#define FLASH_MS 200   // Flash on/off duration in milliseconds

// ===== Motors =====
const int L_PWM_PIN = 9;   // OC1A Left PWM
const int R_PWM_PIN = 10;  // OC1B Right PWM
const int L_DIR_PIN = 7;   // Left DIR
const int R_DIR_PIN = 8;   // Right DIR

// ===== Steering (USER CONFIGURABLE STEERING SPEED BOUNDS) =====
const int SERVO_PIN   = 4;
const int MIN_US      = 1100;
const int MAX_US      = 1700;
const int CENTER_US   = 1400;
const unsigned long DEADMAN_MS = 2000;

ServoTimer2 steer;
int  g_servo_us       = CENTER_US;
bool g_servo_centered = true;
unsigned long g_last_servo_ms = 0;
unsigned long g_last_keepalive_ms = 0;

static inline uint8_t pctToPwm8(int pct) {
  if (pct < 0) pct = 0;
  if (pct > 100) pct = 100;
  // Map 0..100 to 0..255
  return (uint8_t)((pct * 255 + 50) / 100);
}

void setup() {
  steer.attach(SERVO_PIN);
  int cu = CENTER_US;
  if (cu < MIN_US) cu = MIN_US; else if (cu > MAX_US) cu = MAX_US;
  steer.write(cu);
  g_servo_us = cu;
  g_servo_centered = true;
  g_last_servo_ms = millis();
  g_last_keepalive_ms = g_last_servo_ms;

  pinMode(L_DIR_PIN, OUTPUT);
  pinMode(R_DIR_PIN, OUTPUT);
  pinMode(L_PWM_PIN, OUTPUT);
  pinMode(R_PWM_PIN, OUTPUT);

  // Headlight default off
  pinMode(HEADLIGHT_PWM_PIN, OUTPUT);
  analogWrite(HEADLIGHT_PWM_PIN, 0);

  // ---- Timer1 PWM setup ----
  noInterrupts();
  TCCR1A = 0; TCCR1B = 0; TCNT1 = 0;
  ICR1 = MOTOR_PWM_TOP;
  TCCR1A |= (1 << WGM11);
  TCCR1B |= (1 << WGM13) | (1 << WGM12);
  TCCR1A |= (1 << COM1A1) | (1 << COM1B1);
  TCCR1B |= (1 << CS10);  // prescaler = 1
  OCR1A = 0;
  OCR1B = 0;
  interrupts();

  Serial.begin(115200);
}

float clampMotor(float v) {
  // Clamp to defined forward/backward bounds
  if (v >= 0.0f) {
    if (v < MOTOR_FWD_MIN) v = 0.0f;
    else if (v > MOTOR_FWD_MAX) v = MOTOR_FWD_MAX;
  } else {
    if (v > MOTOR_REV_MIN) v = 0.0f;
    else if (v < MOTOR_REV_MAX) v = MOTOR_REV_MAX;
  }
  return v;
}

void loop() {
  static String buf;
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      buf.trim();
      if (buf.length()) {
        char v0 = toupper(buf[0]);
        char v1 = (buf.length() >= 2) ? toupper(buf[1]) : '\0';

        if (v0 == 'S' && v1 == 'T') {
          long us = buf.substring(2).toInt();
          if (us > 0) {
            if (us < MIN_US) us = MIN_US; else if (us > MAX_US) us = MAX_US;
            steer.write((int)us);
            g_servo_us = (int)us;
            g_last_servo_ms = millis();
            g_servo_centered = false;
            Serial.print("OK ST="); Serial.println(g_servo_us);
          } else Serial.println("ERR");

        } else if (v0 == 'T' && v1 == 'K') {
          int sp = buf.indexOf(' ');
          if (sp >= 0) {
            String rest = buf.substring(sp + 1); rest.trim();
            int sp2 = rest.indexOf(' ');
            if (sp2 >= 0) {
              float L = rest.substring(0, sp2).toFloat();
              float R = rest.substring(sp2 + 1).toFloat();

              // Apply bounds
              L = clampMotor(L);
              R = clampMotor(R);

              digitalWrite(L_DIR_PIN, (L >= 0.0f) ? HIGH : LOW);
              digitalWrite(R_DIR_PIN, (R >= 0.0f) ? HIGH : LOW);

              uint16_t top = ICR1;
              OCR1A = (uint16_t)(fabs(L) * top + 0.5f);
              OCR1B = (uint16_t)(fabs(R) * top + 0.5f);

              Serial.print("OK TK L="); Serial.print(L, 3);
              Serial.print(" R=");      Serial.println(R, 3);
            } else Serial.println("ERR");
          } else Serial.println("ERR");

        // ===== New: Headlight brightness command =====
        } else if (v0 == 'H' && v1 == 'L') {
          // Accept "HL 10" or "HL 10%"
          int sp = buf.indexOf(' ');
          if (sp >= 0) {
            String rest = buf.substring(sp + 1); rest.trim();
            // remove optional trailing '%'
            if (rest.endsWith("%")) rest.remove(rest.length() - 1);
            int pct = rest.toInt();
            uint8_t pwm = pctToPwm8(pct);
            analogWrite(HEADLIGHT_PWM_PIN, pwm);
            Serial.print("OK HL="); Serial.print(pct); Serial.println("%");
          } else {
            Serial.println("ERR");
          }

        // ===== New: Headlight flash command =====
        } else if (v0 == 'F' && v1 == 'L') {
          int sp = buf.indexOf(' ');
          if (sp >= 0) {
            int times = buf.substring(sp + 1).toInt();
            if (times <= 0) { Serial.println("ERR"); }
            else {
              // Flash full on then off, times times
              for (int i = 0; i < times; ++i) {
                analogWrite(HEADLIGHT_PWM_PIN, 255);
                delay(FLASH_MS);
                analogWrite(HEADLIGHT_PWM_PIN, 0);
                delay(FLASH_MS);
              }
              Serial.print("OK FL x"); Serial.println(times);
            }
          } else {
            Serial.println("ERR");
          }

        } else Serial.println("ERR");
      }
      buf = "";
    } else {
      buf += c;
      if (buf.length() > 96) buf = "";
    }
  }

  unsigned long now = millis();
  if ((now - g_last_servo_ms) > DEADMAN_MS) {
    if (!g_servo_centered) {
      int cu = CENTER_US;
      if (cu < MIN_US) cu = MIN_US; else if (cu > MAX_US) cu = MAX_US;
      steer.write(cu);
      g_servo_us = cu;
      g_servo_centered = true;
    }
  }

  if ((now - g_last_keepalive_ms) >= 20) {
    steer.write(g_servo_us);
    g_last_keepalive_ms = now;
  }
}

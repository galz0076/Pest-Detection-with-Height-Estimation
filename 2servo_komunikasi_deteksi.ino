#include <Servo.h>

Servo servoVertikal;   // Servo 1 - vertikal (D9)
Servo servoHorizontal; // Servo 2 - horizontal (D10)

void setup() {
  Serial.begin(9600);
  servoVertikal.attach(36);
  servoHorizontal.attach(35);

  // Pastikan servo di tengah sebelum data datang
  servoVertikal.write(90);
  servoHorizontal.write(90);
  delay(500);  // Kasih waktu servo stabil dulu
}

void loop() {
  if (Serial.available() >= 3) {
    byte servoID = Serial.read();
    byte byte1 = Serial.read();
    byte byte2 = Serial.read();
    int pos = (byte1 << 8) | byte2;

    if (pos >= 0 && pos <= 180) {
      if (servoID == 1) {
        servoVertikal.write(pos);
        Serial.print("Servo VERTIKAL ke: ");
        Serial.println(pos);
      } else if (servoID == 2) {
        servoHorizontal.write(pos);
        Serial.print("Servo HORIZONTAL ke: ");
        Serial.println(pos);
      }
    }
  }
}

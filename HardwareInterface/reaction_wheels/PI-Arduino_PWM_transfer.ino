// This code should be pretty simple: read one byte from serial monitor, ensure that it is a valid PWM value, 
// send PWM signal to pin 9

// This program is being used for the jank setup we have where the PI is running the ADCS (but for some reason
// cannot send PWM) and sends the desired PWM signal to an Arduino UNO to output to the motor controller board
// so that the motors can be driven correctly

// Author: Henry Lemersal's Son


int incomingPWM = 0; // Buffer to read in incoming byte of data
int PWMcheck = 0; // var to check what incoming data is
int clearSerial = 0; // clear out any existing junk in serial port


void setup() {
  Serial.begin(115200);
  pinMode(9, OUTPUT); // Pin 9 chosen to output PWM signal

  while (Serial.available() > 0) // if there is anything in the serial port, read it all before main loop
  {
    clearSerial = Serial.read();
  }

}


void loop() {
  while(Serial.available() > 0) // check for information in Serial port
  {
      incomingPWM = Serial.read(); // read correct byte
      analogWrite(9,incomingPWM); // PWM to pin 9

    Serial.print("Setting PWM: ");
    Serial.println(incomingPWM);

    incomingPWM = 0; // reset buffer state
  }
}


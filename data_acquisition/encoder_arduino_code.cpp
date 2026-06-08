#include <Wire.h>
#include <AS5600.h>

// ---------- CONSTANTS ----------
#define TCA9548_ADDR 0x70  // I2C address for TCA9548A multiplexer

// Multiplexer channel assignments
#define I2C_X 1
#define I2C_Y 0
#define I2C_Z 2

// Conversion constants
#define AS5600_RAW_TO_RADIANS (2.0 * M_PI / 4096.0)  // 12-bit sensor = 4096 counts

// For PRUSAXL printer measured and tuned
const float R = 12.7324  / 2.0;      // mm per rad for XY motors
const float P = 8.0 / (4 * M_PI); // mm per rad for Z axis

// ---------- OBJECTS ----------
AS5600 encX;
AS5600 encY;
AS5600 encZ;

// ---------- STATE VARIABLES ----------
float radX = 0, radY = 0, radZ = 0;
float newRadX, newRadY, newRadZ;
float initX, initY, initZ;
float dRadX = 0, dRadY = 0, dRadZ = 0;
float dX = 0, dY = 0, dZ = 0;
float X = 0, Y = 0, Z = 0;
float X0, Y0, Z0;


// ---------- FUNCTION DECLARATIONS ----------
void selectTCAChannel(uint8_t bus);

// ---------- SETUP ----------
void setup() {
  Serial.begin(115200);
  while (!Serial);  // Wait for Serial
  Wire.begin();

  Serial.println("Initializing AS5600 Encoders via TCA9548A...");

  // Initialize encoders one by one
  selectTCAChannel(I2C_X);
  encX.begin();
  encX.setDirection(AS5600_CLOCK_WISE);
  Serial.print("X connected: "); Serial.println(encX.isConnected());
  initX = encX.getCumulativePosition() * AS5600_RAW_TO_RADIANS;

  selectTCAChannel(I2C_Y);
  encY.begin();
  encY.setDirection(AS5600_CLOCK_WISE);
  Serial.print("Y connected: "); Serial.println(encY.isConnected());
  initY = encY.getCumulativePosition() * AS5600_RAW_TO_RADIANS;

  selectTCAChannel(I2C_Z);
  encZ.begin();
  encZ.setDirection(AS5600_CLOCK_WISE);
  Serial.print("Z connected: "); Serial.println(encZ.isConnected());
  initZ = encZ.getCumulativePosition() * AS5600_RAW_TO_RADIANS;

  // Initialize position
  X = 30;
  Y = 30;
  Z = 2;

  X0 = X, Y0 = Y, Z0 = Z;

Serial.println("Encoder setup complete.");
  delay(2000);  // Let things stabilize
}

// ---------- MAIN LOOP ----------
void loop() {
  static uint32_t lastTime = 0;
  const uint32_t interval = 13.33; // ms between readings (~75 Hz)

  if (millis() - lastTime >= interval) {
    lastTime = millis();

    // Read all three encoders sequentially through multiplexer
    selectTCAChannel(I2C_X);
    newRadX = encX.getCumulativePosition() * AS5600_RAW_TO_RADIANS - initX;
    dRadX = newRadX - radX;
    radX = newRadX;

    selectTCAChannel(I2C_Y);
    newRadY = encY.getCumulativePosition() * AS5600_RAW_TO_RADIANS - initY;
    dRadY = newRadY - radY;
    radY = newRadY;

    selectTCAChannel(I2C_Z);
    newRadZ = encZ.getCumulativePosition() * AS5600_RAW_TO_RADIANS - initZ;
    dRadZ = newRadZ - radZ;
    radZ = newRadZ;

    // --- POSITION COMPUTATION
    // For PrusaXL coreXY:
    X += (dRadX + dRadY) * R /2;
    Y -= (dRadX - dRadY) * R /2;
    Z -= dRadZ * P;

    //Change in position computation
    dX = X - X0;
    X0 = X;

    dY = Y - Y0;
    Y0 = Y;

    dZ = Z - Z0;
    Z0 = Z;

    // --- SERIAL OUTPUT ---
    Serial.print(millis());
    Serial.print(",");
    Serial.print(X, 3);
    Serial.print(",");
    Serial.print(Y, 3);
    Serial.print(",");
    Serial.print(Z, 3);
    Serial.print(",");
    Serial.print(dX, 3);
    Serial.print(",");
    Serial.print(dY, 3);
    Serial.print(",");
    Serial.print(dZ, 3);
    Serial.println();
  }
}

// ---------- TCA9548 CHANNEL SELECT ----------
void selectTCAChannel(uint8_t bus) {
  Wire.beginTransmission(TCA9548_ADDR);
  Wire.write(1 << bus);
  Wire.endTransmission();
  delayMicroseconds(150); // Short pause to ensure bus switching settles
}

"""
Signal Handler — Send Inspection Verdicts to PLC via Modbus

Writes the final inspection result (PASS / REWORK / REJECT) to a single
Modbus coil on the PLC.  Supports both Modbus RTU (serial) and Modbus TCP
(Ethernet) connections.

Coil encoding (configurable):
    PASS   → coil ON  (True)
    REJECT → coil OFF (False)
    REWORK → coil OFF (False)   # treated same as REJECT by default

All communication parameters are editable in the CONFIGURATION section below
or can be overridden at runtime via constructor arguments.

Usage (standalone test):
    python rework/signal_handler.py

Dependencies:
    pip install pymodbus

Author: GitHub Copilot
Date:   February 20, 2026
"""

from enum import Enum
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these values to match your PLC setup
# ═══════════════════════════════════════════════════════════════════════════

# ── Connection mode ───────────────────────────────────────────────────────
#   "rtu"  → Modbus RTU over serial (RS-232 / RS-485)
#   "tcp"  → Modbus TCP over Ethernet
CONNECTION_MODE = "rtu"

# ── Modbus RTU (serial) settings ──────────────────────────────────────────
SERIAL_PORT   = "COM3"        # Serial / COM port
BAUDRATE      = 9600          # Baud rate (common: 9600, 19200, 38400, 115200)
PARITY        = "N"           # "N" = None, "E" = Even, "O" = Odd
STOPBITS      = 1             # Stop bits (1 or 2)
BYTESIZE      = 8             # Data bits (7 or 8)
SERIAL_TIMEOUT = 1.0          # Read timeout in seconds

# ── Modbus TCP (Ethernet) settings ────────────────────────────────────────
TCP_HOST      = "192.168.1.10"   # PLC IP address
TCP_PORT      = 502              # Modbus TCP port (default 502)
TCP_TIMEOUT   = 3.0              # Connection timeout in seconds

# ── Modbus addressing ────────────────────────────────────────────────────
SLAVE_ID      = 1             # Modbus slave / unit ID
COIL_ADDRESS  = 0             # Coil address to write the result to

# ── Verdict → coil value mapping ─────────────────────────────────────────
#   True  = coil ON,  False = coil OFF
VERDICT_MAP = {
    "PASS":    True,
    "REWORK":  False,
    "REJECT":  False,
}


# ═══════════════════════════════════════════════════════════════════════════
#  Parity helper
# ═══════════════════════════════════════════════════════════════════════════

_PARITY_LOOKUP = {
    "N": "N",
    "E": "E",
    "O": "O",
    "NONE": "N",
    "EVEN": "E",
    "ODD":  "O",
}


def _resolve_parity(value: str) -> str:
    """Normalise a parity string to the single-char code pymodbus expects."""
    return _PARITY_LOOKUP.get(value.upper().strip(), "N")


# ═══════════════════════════════════════════════════════════════════════════
#  SignalHandler class
# ═══════════════════════════════════════════════════════════════════════════

class SignalHandler:
    """Send inspection verdicts to a PLC over Modbus (RTU or TCP).

    Parameters
    ----------
    mode : str
        ``"rtu"`` for serial or ``"tcp"`` for Ethernet.  Falls back to
        the module-level ``CONNECTION_MODE`` constant.
    serial_port, baudrate, parity, stopbits, bytesize, serial_timeout :
        Serial parameters (RTU mode only).  Defaults come from the
        module-level constants above.
    tcp_host, tcp_port, tcp_timeout :
        Ethernet parameters (TCP mode only).
    slave_id : int
        Modbus slave / unit address.
    coil_address : int
        Address of the coil that receives the verdict.
    verdict_map : dict
        Maps verdict strings to bool coil values.
    auto_connect : bool
        If ``True`` (default), connect immediately on construction.
    """

    def __init__(
        self,
        mode: Optional[str] = None,
        # RTU
        serial_port: Optional[str] = None,
        baudrate: Optional[int] = None,
        parity: Optional[str] = None,
        stopbits: Optional[int] = None,
        bytesize: Optional[int] = None,
        serial_timeout: Optional[float] = None,
        # TCP
        tcp_host: Optional[str] = None,
        tcp_port: Optional[int] = None,
        tcp_timeout: Optional[float] = None,
        # Modbus
        slave_id: Optional[int] = None,
        coil_address: Optional[int] = None,
        verdict_map: Optional[dict] = None,
        auto_connect: bool = True,
    ):
        self.mode = (mode or CONNECTION_MODE).lower().strip()

        # RTU settings
        self.serial_port    = serial_port    or SERIAL_PORT
        self.baudrate       = baudrate       or BAUDRATE
        self.parity         = _resolve_parity(parity or PARITY)
        self.stopbits       = stopbits       or STOPBITS
        self.bytesize       = bytesize       or BYTESIZE
        self.serial_timeout = serial_timeout if serial_timeout is not None else SERIAL_TIMEOUT

        # TCP settings
        self.tcp_host    = tcp_host    or TCP_HOST
        self.tcp_port    = tcp_port    or TCP_PORT
        self.tcp_timeout = tcp_timeout if tcp_timeout is not None else TCP_TIMEOUT

        # Modbus addressing
        self.slave_id     = slave_id     if slave_id     is not None else SLAVE_ID
        self.coil_address = coil_address if coil_address is not None else COIL_ADDRESS

        # Verdict mapping
        self.verdict_map = verdict_map or dict(VERDICT_MAP)

        # Client handle (created on connect)
        self._client = None

        if auto_connect:
            self.connect()

    # ── Connection management ─────────────────────────────────────────────

    def connect(self) -> bool:
        """Open the Modbus connection.  Returns True on success."""
        if self._client is not None:
            return True  # already connected

        try:
            if self.mode == "rtu":
                from pymodbus.client import ModbusSerialClient
                self._client = ModbusSerialClient(
                    port=self.serial_port,
                    baudrate=self.baudrate,
                    parity=self.parity,
                    stopbits=self.stopbits,
                    bytesize=self.bytesize,
                    timeout=self.serial_timeout,
                )
            elif self.mode == "tcp":
                from pymodbus.client import ModbusTcpClient
                self._client = ModbusTcpClient(
                    host=self.tcp_host,
                    port=self.tcp_port,
                    timeout=self.tcp_timeout,
                )
            else:
                raise ValueError(f"Unknown mode '{self.mode}'. Use 'rtu' or 'tcp'.")

            connected = self._client.connect()
            if not connected:
                print(f"[SignalHandler] ⚠ Failed to connect ({self.mode.upper()}).")
                self._client = None
                return False

            print(f"[SignalHandler] ✓ Connected via {self.mode.upper()}"
                  f" ({'port=' + self.serial_port if self.mode == 'rtu' else 'host=' + self.tcp_host})")
            return True

        except Exception as exc:
            print(f"[SignalHandler] ⚠ Connection error: {exc}")
            self._client = None
            return False

    def disconnect(self):
        """Close the Modbus connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            print("[SignalHandler] Disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    # ── Verdict sending ───────────────────────────────────────────────────

    def send_verdict(self, verdict: str) -> bool:
        """Write the inspection verdict to the PLC coil.

        Parameters
        ----------
        verdict : str
            One of ``"PASS"``, ``"REWORK"``, or ``"REJECT"``
            (case-insensitive).

        Returns
        -------
        bool
            ``True`` if the write succeeded, ``False`` otherwise.
        """
        verdict_upper = verdict.strip().upper()

        if verdict_upper not in self.verdict_map:
            print(f"[SignalHandler] ⚠ Unknown verdict '{verdict}'. "
                  f"Expected one of {list(self.verdict_map.keys())}.")
            return False

        coil_value = self.verdict_map[verdict_upper]

        if self._client is None:
            print("[SignalHandler] ⚠ Not connected — attempting reconnect …")
            if not self.connect():
                return False

        try:
            result = self._client.write_coil(
                address=self.coil_address,
                value=coil_value,
                slave=self.slave_id,
            )

            if result.isError():
                print(f"[SignalHandler] ⚠ Modbus write error: {result}")
                return False

            print(f"[SignalHandler] ✓ Sent {verdict_upper} → "
                  f"coil {self.coil_address} = {coil_value}")
            return True

        except Exception as exc:
            print(f"[SignalHandler] ⚠ Write failed: {exc}")
            return False

    # ── Convenience wrappers ──────────────────────────────────────────────

    def send_pass(self) -> bool:
        return self.send_verdict("PASS")

    def send_rework(self) -> bool:
        return self.send_verdict("REWORK")

    def send_reject(self) -> bool:
        return self.send_verdict("REJECT")

    # ── Read-back (optional verification) ─────────────────────────────────

    def read_coil(self) -> Optional[bool]:
        """Read the current coil value back from the PLC (for verification).

        Returns ``None`` on failure.
        """
        if self._client is None:
            print("[SignalHandler] ⚠ Not connected.")
            return None
        try:
            result = self._client.read_coils(
                address=self.coil_address,
                count=1,
                slave=self.slave_id,
            )
            if result.isError():
                print(f"[SignalHandler] ⚠ Read error: {result}")
                return None
            return result.bits[0]
        except Exception as exc:
            print(f"[SignalHandler] ⚠ Read failed: {exc}")
            return None

    # ── Context manager support ───────────────────────────────────────────

    def __enter__(self):
        if not self.is_connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    def __repr__(self):
        mode_info = (f"port={self.serial_port}, baud={self.baudrate}"
                     if self.mode == "rtu"
                     else f"host={self.tcp_host}:{self.tcp_port}")
        return (f"SignalHandler(mode={self.mode!r}, {mode_info}, "
                f"slave={self.slave_id}, coil={self.coil_address})")


# ═══════════════════════════════════════════════════════════════════════════
#  Standalone test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Signal Handler — standalone test")
    print("=" * 60)
    print(f"  Mode         : {CONNECTION_MODE.upper()}")
    if CONNECTION_MODE.lower() == "rtu":
        print(f"  Serial Port  : {SERIAL_PORT}")
        print(f"  Baud Rate    : {BAUDRATE}")
        print(f"  Parity       : {PARITY}")
        print(f"  Stop Bits    : {STOPBITS}")
        print(f"  Byte Size    : {BYTESIZE}")
    else:
        print(f"  TCP Host     : {TCP_HOST}")
        print(f"  TCP Port     : {TCP_PORT}")
    print(f"  Slave ID     : {SLAVE_ID}")
    print(f"  Coil Address : {COIL_ADDRESS}")
    print(f"  Verdict Map  : {VERDICT_MAP}")
    print("=" * 60)

    handler = SignalHandler(auto_connect=False)
    print(f"\nHandler: {handler}")

    print("\nAttempting connection …")
    if handler.connect():
        for verdict in ["PASS", "REWORK", "REJECT"]:
            ok = handler.send_verdict(verdict)
            readback = handler.read_coil()
            print(f"  {verdict:8s} → write={'OK' if ok else 'FAIL'}, "
                  f"readback={readback}")
        handler.disconnect()
    else:
        print("Could not connect — check your settings above.")

"""prana — health check and metrics."""
import time, platform, sys
from typing import Dict, Any

_start_time = time.time()

def get_health(metrics: Dict[str, Any] = None) -> Dict[str, Any]:
    """Return service health status."""
    return {
        "service": "prana",
        "status": "ok",
        "version": "0.1.0",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "python_version": sys.version.split()[0],
        "platform": platform.system(),
        "metrics": metrics or {},
    }

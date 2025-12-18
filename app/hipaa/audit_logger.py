"""Immutable audit logging for HIPAA compliance with blockchain-style integrity."""

import json
import logging
import hashlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class HIPAAAuditLogger:
    """Immutable audit logging system for HIPAA compliance."""

    def __init__(self, log_dir: str = "./audit_logs"):
        """
        Initialize audit logger.

        Args:
            log_dir: Directory to store audit logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Ensure log directory has secure permissions
        os.chmod(self.log_dir, 0o700)

        # Current log file based on date
        self.current_date = datetime.now(timezone.utc).date()
        self.log_file = self.log_dir / f"audit_{self.current_date.isoformat()}.log"

        # Initialize logger
        self._setup_logger()

        # Get previous hash for chaining
        self.previous_hash = self._get_last_hash()

        # Performance monitoring
        self.entries_today = 0
        self.max_entries_per_day = 100000  # Prevent log file from growing too large

    def _setup_logger(self):
        """Setup logger with append-only file handler."""
        # Create logger if it doesn't exist
        self.logger = logging.getLogger(f"hipaa_audit_{id(self)}")
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create file handler with append-only mode
        handler = logging.FileHandler(
            self.log_file,
            mode='a',
            encoding='utf-8'
        )

        # Set formatter (JSON format)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False

    def _get_last_hash(self) -> str:
        """Get hash of last audit entry for blockchain-style chaining."""
        # Try current day's log first
        last_hash = self._get_last_hash_from_file(self.log_file)
        if last_hash:
            return last_hash

        # Check previous day's logs
        for days_back in range(1, 7):  # Check up to 7 days back
            date = self.current_date.isoformat()
            log_file = self.log_dir / f"audit_{date}.log"
            last_hash = self._get_last_hash_from_file(log_file)
            if last_hash:
                return last_hash

        # Genesis hash if no previous entries
        return "0" * 64

    def _get_last_hash_from_file(self, log_file: Path) -> Optional[str]:
        """Get last hash from specific log file."""
        if not log_file.exists():
            return None

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    # Get last non-empty line
                    for line in reversed(lines):
                        line = line.strip()
                        if line:
                            entry = json.loads(line)
                            return entry.get("hash", "0" * 64)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read last hash from {log_file}: {e}")

        return None

    def _compute_hash(self, entry: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of entry + previous hash."""
        # Remove hash if present (computing hash of hash would be circular)
        entry_copy = entry.copy()
        entry_copy.pop("hash", None)

        # Convert to sorted JSON string for consistent ordering
        entry_str = json.dumps(entry_copy, sort_keys=True, separators=(',', ':'))

        # Combine with previous hash
        combined = f"{self.previous_hash}{entry_str}"

        # Compute SHA-256 hash
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def _write_entry(self, entry: Dict[str, Any]) -> str:
        """Write audit entry to log file."""
        try:
            # Check if we need to rotate to a new day
            current_date = datetime.now(timezone.utc).date()
            if current_date != self.current_date:
                self._rotate_log()

            # Compute and add hash
            entry["hash"] = self._compute_hash(entry)

            # Write entry
            entry_json = json.dumps(entry, separators=(',', ':'))
            self.logger.info(entry_json)

            # Update state
            self.previous_hash = entry["hash"]
            self.entries_today += 1

            return entry["hash"]

        except Exception as e:
            logger.error(f"Failed to write audit entry: {e}")
            raise

    def _rotate_log(self):
        """Rotate log file for new day."""
        self.current_date = datetime.now(timezone.utc).date()
        self.log_file = self.log_dir / f"audit_{self.current_date.isoformat()}.log"

        # Reset logger with new file
        self._setup_logger()

        # Reset previous hash for new day (but keep last hash from previous day)
        self.entries_today = 0

    def log_phi_access(
        self,
        user_id: str,
        patient_id: str,
        action: str,
        resource: str,
        result: str = "success",
        **kwargs
    ):
        """
        Log PHI access with complete audit trail.

        Args:
            user_id: User identifier
            patient_id: Patient identifier (encrypted)
            action: Action performed (read, write, delete, etc.)
            resource: Resource being accessed (consultation_id, file_name, etc.)
            result: Result of access (success, failure, denied)
            **kwargs: Additional metadata
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "PHI_ACCESS",
            "user_id": user_id,
            "patient_id": patient_id,
            "action": action,
            "resource": resource,
            "result": result,
            "ip_address": kwargs.get("ip_address"),
            "user_agent": kwargs.get("user_agent"),
            "session_id": kwargs.get("session_id"),
            "access_reason": kwargs.get("access_reason"),
            "previous_hash": self.previous_hash
        }

        # Add any additional fields
        for key, value in kwargs.items():
            if key not in entry:
                entry[key] = value

        self._write_entry(entry)

    def log_system_event(
        self,
        event_type: str,
        user_id: str = None,
        details: Dict[str, Any] = None
    ):
        """
        Log system events.

        Args:
            event_type: Type of system event
            user_id: User identifier (if applicable)
            details: Event details dictionary
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details or {},
            "previous_hash": self.previous_hash
        }

        self._write_entry(entry)

    def log_authentication_event(
        self,
        user_id: str,
        event_type: str,  # login, logout, failed_login, token_expired
        ip_address: str = None,
        user_agent: str = None,
        success: bool = True
    ):
        """
        Log authentication events.

        Args:
            user_id: User identifier
            event_type: Authentication event type
            ip_address: Client IP address
            user_agent: Client user agent
            success: Whether authentication was successful
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "AUTHENTICATION",
            "user_id": user_id,
            "auth_event": event_type,
            "success": success,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "previous_hash": self.previous_hash
        }

        self._write_entry(entry)

    def log_data_modification(
        self,
        user_id: str,
        action: str,  # create, update, delete
        resource_type: str,  # consultation, patient, entity, etc.
        resource_id: str,
        changes: Dict[str, Any] = None
    ):
        """
        Log data modification events.

        Args:
            user_id: User identifier
            action: Modification action
            resource_type: Type of resource modified
            resource_id: Resource identifier
            changes: Dictionary of changes made
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "DATA_MODIFICATION",
            "user_id": user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "changes": changes or {},
            "previous_hash": self.previous_hash
        }

        self._write_entry(entry)

    def log_export_event(
        self,
        user_id: str,
        export_type: str,
        record_count: int,
        destination: str,
        patient_ids: List[str] = None
    ):
        """
        Log data export events.

        Args:
            user_id: User identifier
            export_type: Type of export (PDF, CSV, JSON, etc.)
            record_count: Number of records exported
            destination: Export destination (email, download, etc.)
            patient_ids: List of patient IDs involved
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "DATA_EXPORT",
            "user_id": user_id,
            "export_type": export_type,
            "record_count": record_count,
            "destination": destination,
            "patient_ids": patient_ids or [],
            "previous_hash": self.previous_hash
        }

        self._write_entry(entry)

    def log_breach_attempt(
        self,
        description: str,
        ip_address: str,
        user_id: str = None,
        details: Dict[str, Any] = None
    ):
        """
        Log security breach attempts.

        Args:
            description: Description of breach attempt
            ip_address: Source IP address
            user_id: User identifier (if applicable)
            details: Additional details about the attempt
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "SECURITY_BREACH_ATTEMPT",
            "description": description,
            "ip_address": ip_address,
            "user_id": user_id,
            "details": details or {},
            "severity": "high",
            "previous_hash": self.previous_hash
        }

        self._write_entry(entry)

    def search_logs(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        user_id: str = None,
        patient_id: str = None,
        event_type: str = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Search audit logs with filters.

        Args:
            start_date: Start date for search
            end_date: End date for search
            user_id: Filter by user ID
            patient_id: Filter by patient ID
            event_type: Filter by event type
            limit: Maximum number of results

        Returns:
            List of matching audit entries
        """
        results = []

        # Determine date range
        if not start_date:
            start_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        if not end_date:
            end_date = datetime.now(timezone.utc)

        # Search log files in date range
        current_date = start_date.date()
        while current_date <= end_date.date():
            log_file = self.log_dir / f"audit_{current_date.isoformat()}.log"
            if log_file.exists():
                results.extend(self._search_log_file(
                    log_file, start_date, end_date, user_id, patient_id, event_type
                ))

            current_date = datetime.combine(
                current_date,
                datetime.min.time()
            ) + timedelta(days=1)

            if len(results) >= limit:
                break

        return results[:limit]

    def _search_log_file(
        self,
        log_file: Path,
        start_date: datetime,
        end_date: datetime,
        user_id: str = None,
        patient_id: str = None,
        event_type: str = None
    ) -> List[Dict[str, Any]]:
        """Search a specific log file."""
        results = []

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line.strip())

                        # Parse timestamp
                        timestamp = datetime.fromisoformat(entry["timestamp"])

                        # Check date range
                        if timestamp < start_date or timestamp > end_date:
                            continue

                        # Apply filters
                        if user_id and entry.get("user_id") != user_id:
                            continue
                        if patient_id and entry.get("patient_id") != patient_id:
                            continue
                        if event_type and entry.get("event_type") != event_type:
                            continue

                        # Add line number for reference
                        entry["line_number"] = line_num
                        entry["log_file"] = str(log_file)

                        results.append(entry)

                    except json.JSONDecodeError:
                        continue  # Skip malformed lines

        except IOError as e:
            logger.warning(f"Failed to search log file {log_file}: {e}")

        return results

    def verify_audit_trail(self) -> Dict[str, Any]:
        """
        Verify integrity of entire audit trail.

        Returns:
            Dictionary with verification results
        """
        verification_result = {
            "verified": True,
            "errors": [],
            "total_entries": 0,
            "date_range": None,
            "first_entry_hash": None,
            "last_entry_hash": None
        }

        try:
            # Get all log files
            log_files = sorted(self.log_dir.glob("audit_*.log"))

            if not log_files:
                verification_result["verified"] = False
                verification_result["errors"].append("No audit log files found")
                return verification_result

            previous_hash = "0" * 64
            entry_count = 0
            first_date = None
            last_date = None

            for log_file in log_files:
                file_errors = self._verify_log_file(log_file, previous_hash)
                if file_errors:
                    verification_result["verified"] = False
                    verification_result["errors"].extend(file_errors)
                    break

                # Update stats
                file_entries = self._count_file_entries(log_file)
                entry_count += file_entries

                # Get date range
                if file_entries > 0:
                    file_date = self._extract_date_from_filename(log_file)
                    if not first_date:
                        first_date = file_date
                    last_date = file_date

                    # Update previous hash for next file
                    previous_hash = self._get_last_hash_from_file(log_file) or previous_hash

            verification_result.update({
                "total_entries": entry_count,
                "date_range": {
                    "start": first_date.isoformat() if first_date else None,
                    "end": last_date.isoformat() if last_date else None
                }
            })

        except Exception as e:
            verification_result["verified"] = False
            verification_result["errors"].append(f"Verification failed: {str(e)}")

        return verification_result

    def _verify_log_file(self, log_file: Path, expected_previous_hash: str) -> List[str]:
        """Verify integrity of a single log file."""
        errors = []

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                previous_hash = expected_previous_hash
                line_number = 0

                for line in f:
                    line_number += 1
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)

                        # Check previous hash
                        stored_previous = entry.get("previous_hash")
                        if stored_previous != previous_hash:
                            errors.append(
                                f"Hash chain broken in {log_file.name} line {line_number}: "
                                f"expected {previous_hash}, got {stored_previous}"
                            )

                        # Verify current hash
                        stored_hash = entry.pop("hash", None)
                        if stored_hash:
                            computed_hash = self._compute_hash(entry)
                            if stored_hash != computed_hash:
                                errors.append(
                                    f"Hash verification failed in {log_file.name} line {line_number}: "
                                    f"stored {stored_hash}, computed {computed_hash}"
                                )
                            previous_hash = stored_hash
                        else:
                            errors.append(
                                f"Missing hash in {log_file.name} line {line_number}"
                            )

                    except json.JSONDecodeError:
                        errors.append(f"Invalid JSON in {log_file.name} line {line_number}")

        except IOError as e:
            errors.append(f"Failed to read {log_file}: {str(e)}")

        return errors

    def _count_file_entries(self, log_file: Path) -> int:
        """Count valid entries in a log file."""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())
        except IOError:
            return 0

    def _extract_date_from_filename(self, log_file: Path) -> datetime:
        """Extract date from log filename."""
        filename = log_file.stem  # Remove .log extension
        date_str = filename.replace("audit_", "")
        return datetime.fromisoformat(date_str).date()

    @contextmanager
    def audit_context(self, user_id: str, operation: str, resource: str):
        """
        Context manager for automatic audit logging.

        Args:
            user_id: User identifier
            operation: Operation being performed
            resource: Resource being operated on
        """
        start_time = datetime.now(timezone.utc)
        entry_hash = None

        try:
            # Log operation start
            entry_hash = self._write_entry({
                "timestamp": start_time.isoformat(),
                "event_type": "OPERATION_START",
                "user_id": user_id,
                "operation": operation,
                "resource": resource,
                "status": "started",
                "previous_hash": self.previous_hash
            })

            yield

            # Log operation success
            end_time = datetime.now(timezone.utc)
            duration_ms = (end_time - start_time).total_seconds() * 1000

            self._write_entry({
                "timestamp": end_time.isoformat(),
                "event_type": "OPERATION_COMPLETE",
                "user_id": user_id,
                "operation": operation,
                "resource": resource,
                "status": "success",
                "duration_ms": duration_ms,
                "started_entry_hash": entry_hash,
                "previous_hash": self.previous_hash
            })

        except Exception as e:
            # Log operation failure
            end_time = datetime.now(timezone.utc)
            duration_ms = (end_time - start_time).total_seconds() * 1000

            self._write_entry({
                "timestamp": end_time.isoformat(),
                "event_type": "OPERATION_FAILED",
                "user_id": user_id,
                "operation": operation,
                "resource": resource,
                "status": "failed",
                "error": str(e),
                "duration_ms": duration_ms,
                "started_entry_hash": entry_hash,
                "previous_hash": self.previous_hash
            })

            raise
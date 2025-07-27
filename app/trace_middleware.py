"""
Middleware for adding trace ID to requests for easier debugging.
"""

import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from contextvars import ContextVar
from typing import Callable

# Context variable to store the current trace ID
trace_context: ContextVar[str] = ContextVar('trace_id', default='')


class TraceMiddleware(BaseHTTPMiddleware):
    """Middleware to add trace ID to each request."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate or extract trace ID
        trace_id = request.headers.get('X-Trace-ID', str(uuid.uuid4()))
        
        # Set trace ID in context
        trace_context.set(trace_id)
        
        # Process request
        response = await call_next(request)
        
        # Add trace ID to response headers
        response.headers['X-Trace-ID'] = trace_id
        
        return response


def get_trace_id() -> str:
    """Get the current trace ID from context."""
    return trace_context.get('')
"""HTTP boundary for the optional integrated prefill decision service."""

import json

from fastapi.responses import ORJSONResponse
from pydantic import ValidationError


async def handle_systemone(request, service):
    # The optional adapter package is imported only in decision-serving mode.
    from jev_adapter.backend import AdapterError
    from jev_adapter.protocol import SystemOneRequest
    from jev_adapter.server import score_http

    try:
        body = SystemOneRequest.model_validate(await request.json())
        return ORJSONResponse(await score_http(service, body, request))
    except ValidationError as error:
        first = error.errors()[0]
        return ORJSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": first["msg"],
                    "field": ".".join(str(part) for part in first["loc"]),
                }
            },
        )
    except json.JSONDecodeError:
        return ORJSONResponse(
            status_code=400,
            content={
                "error": {"code": "invalid_json", "message": "Invalid JSON body."}
            },
        )
    except AdapterError as error:
        return ORJSONResponse(
            status_code=error.status,
            content={
                "error": {
                    "code": error.code,
                    "message": str(error),
                    "field": error.field,
                }
            },
        )

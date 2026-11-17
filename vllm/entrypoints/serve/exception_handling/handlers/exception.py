# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import Request
from starlette.responses import JSONResponse

from vllm.logger import init_logger

from ..error_response import create_error_response

logger = init_logger(__name__)

# Starlette's JSONResponse refuses NaN and infinite floats with a ValueError that
# starts with this. A route that renders the engine's output raises it when the
# engine computed one, such as a logprob from NaN logits.
NON_FINITE_FLOAT_ERROR = "Out of range float values are not JSON compliant"


def is_non_finite_output(exc: Exception) -> bool:
    """Whether `exc` is the JSON encoder refusing a NaN or infinite float."""
    return isinstance(exc, ValueError) and str(exc).startswith(NON_FINITE_FLOAT_ERROR)


async def exception_handler(req: Request, exc: Exception):
    request_id = (
        req.state.request_metadata.request_id
        if hasattr(req.state, "request_metadata")
        else None
    )
    if is_non_finite_output(exc):
        # The engine's output failed, not the request, so this logs either way.
        logger.error(
            "Request %s: the response holds a NaN or infinite value that the "
            "engine computed: %s",
            request_id,
            exc,
        )
    elif req.app.state.args.log_error_stack:
        logger.error(
            "Exception caught. Request id: %s: %s: %s",
            request_id,
            type(exc).__name__,
            exc,
        )

    err = create_error_response(exc)
    return JSONResponse(err.model_dump(), status_code=err.error.code)

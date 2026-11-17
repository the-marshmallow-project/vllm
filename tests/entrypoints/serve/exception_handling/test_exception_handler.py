# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fallback exception handler logs why a request failed, and always logs a
response that the JSON encoder refused for a NaN or infinite value."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from vllm.entrypoints.serve.exception_handling.handlers import exception


def make_client(log_error_stack: bool) -> TestClient:
    app = FastAPI()
    app.state.args = SimpleNamespace(log_error_stack=log_error_stack)
    app.exception_handler(ValueError)(exception.exception_handler)

    @app.get("/nan")
    def nan():
        return JSONResponse({"logprob": float("nan")})

    @app.get("/invalid")
    def invalid():
        raise ValueError("max_tokens must be at least 1")

    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("log_error_stack", [False, True])
def test_nan_output_is_logged(log_error_stack: bool):
    with patch.object(exception.logger, "error") as error:
        response = make_client(log_error_stack).get("/nan")

    assert response.status_code == 400
    assert exception.NON_FINITE_FLOAT_ERROR in response.json()["error"]["message"]
    error.assert_called_once()
    assert "NaN or infinite value" in error.call_args.args[0]


def test_other_errors_log_their_reason():
    with patch.object(exception.logger, "error") as error:
        response = make_client(log_error_stack=True).get("/invalid")

    assert response.status_code == 400
    error.assert_called_once()
    _, _, name, logged = error.call_args.args
    assert name == "ValueError"
    assert str(logged) == "max_tokens must be at least 1"


def test_other_errors_stay_quiet_without_log_error_stack():
    with patch.object(exception.logger, "error") as error:
        response = make_client(log_error_stack=False).get("/invalid")

    assert response.status_code == 400
    error.assert_not_called()

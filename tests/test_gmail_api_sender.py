import base64
from io import BytesIO
import json
from email import message_from_bytes
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest

from gmail_api_sender import SEND_URL, TOKEN_URL, send_gmail_api_email


def _response(payload):
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(payload).encode("utf-8")
    return response


@patch("gmail_api_sender.urllib.request.urlopen")
def test_sends_message_via_gmail_api(mock_urlopen):
    mock_urlopen.side_effect = [
        _response({"access_token": "access-token"}),
        _response({"id": "message-id"}),
    ]

    send_gmail_api_email(
        "student@example.com",
        "Progress update",
        "Please contact the support team.",
        "client-id",
        "client-secret",
        "refresh-token",
        "teacher@example.com",
    )

    token_request, send_request = [call.args[0] for call in mock_urlopen.call_args_list]
    assert token_request.full_url == TOKEN_URL
    assert send_request.full_url == SEND_URL
    assert send_request.get_header("Authorization") == "Bearer access-token"
    payload = json.loads(send_request.data.decode("utf-8"))
    raw_message = base64.urlsafe_b64decode(payload["raw"])
    message = message_from_bytes(raw_message)
    assert message["To"] == "student@example.com"
    assert message["From"] == "teacher@example.com"
    assert message["Subject"] == "Progress update"
    assert message.get_payload(decode=True).decode("utf-8").strip() == (
        "Please contact the support team."
    )


@patch("gmail_api_sender.urllib.request.urlopen")
def test_reports_invalid_refresh_token(mock_urlopen):
    mock_urlopen.side_effect = HTTPError(
        TOKEN_URL,
        400,
        "Bad Request",
        {},
        BytesIO(b'{"error":"invalid_grant","error_description":"Bad Request"}'),
    )

    with pytest.raises(RuntimeError, match="OAuth token request failed \\(HTTP 400\\)"):
        send_gmail_api_email(
            "student@example.com",
            "Subject",
            "Body",
            "client-id",
            "client-secret",
            "expired-refresh-token",
            "teacher@example.com",
        )

    assert mock_urlopen.call_count == 1


@patch("gmail_api_sender.urllib.request.urlopen")
def test_reports_gmail_api_send_error(mock_urlopen):
    mock_urlopen.side_effect = [
        _response({"access_token": "access-token"}),
        HTTPError(
            SEND_URL,
            403,
            "Forbidden",
            {},
            BytesIO(b'{"error":{"message":"Insufficient Permission"}}'),
        ),
    ]

    with pytest.raises(RuntimeError, match="Gmail send request failed \\(HTTP 403\\)"):
        send_gmail_api_email(
            "student@example.com",
            "Subject",
            "Body",
            "client-id",
            "client-secret",
            "refresh-token",
            "teacher@example.com",
        )

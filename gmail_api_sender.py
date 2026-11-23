import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from email.message import EmailMessage


TOKEN_URL = "https://oauth2.googleapis.com/token"
SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"


def _request_json(request, service):
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}

        details = payload.get("error", {})
        if isinstance(details, dict):
            description = details.get("message", "")
        else:
            description = payload.get("error_description", "") or str(details)
        suffix = f": {description}" if description else ""
        raise RuntimeError(
            f"Google {service} request failed (HTTP {exc.code}){suffix}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to Google's {service} service.") from exc


def send_gmail_api_email(
    recipient, subject, body, client_id, client_secret, refresh_token, sender
):
    token_request = urllib.request.Request(
        TOKEN_URL,
        data=urllib.parse.urlencode(
            {
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }
        ).encode("ascii"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    token_response = _request_json(token_request, "OAuth token")
    access_token = token_response.get("access_token")
    if not access_token:
        raise RuntimeError("Google OAuth response did not include an access token.")

    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")

    send_request = urllib.request.Request(
        SEND_URL,
        data=json.dumps({"raw": encoded_message}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    _request_json(send_request, "Gmail send")

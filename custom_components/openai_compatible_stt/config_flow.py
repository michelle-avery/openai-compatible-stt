"""Config flow for OpenAI Compatible speech-to-text custom component."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import voluptuous as vol

from homeassistant import data_entry_flow
from homeassistant.config_entries import ConfigFlow
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.selector import selector

from .const import CONF_API_KEY, CONF_MODEL, CONF_URL, DOMAIN, MODELS, UNIQUE_ID

_LOGGER = logging.getLogger(__name__)


def generate_unique_id(user_input: dict) -> str:
    """Generate a unique id from user input."""
    url = urlparse(user_input[CONF_URL])
    return f"{url.hostname}_{user_input[CONF_MODEL]}"


async def validate_user_input(user_input: dict):
    """Validate user input fields."""
    if user_input.get(CONF_MODEL) is None:
        raise ValueError("Model is required")


class OpenAISTTConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenAI Compatible STT."""

    VERSION = 1
    data_schema = vol.Schema(
        {
            vol.Optional(CONF_API_KEY): str,
            vol.Optional(CONF_URL, default="https://api.openai.com/v1/"): str,
            vol.Required(CONF_MODEL, default="whisper-1"): selector(
                {
                    "select": {
                        "options": MODELS,
                        "mode": "dropdown",
                        "sort": True,
                        "custom_value": True,
                    }
                }
            ),
        }
    )

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        """Handle the initial step."""
        errors = {}
        if user_input is not None:
            try:
                await validate_user_input(user_input)
                unique_id = generate_unique_id(user_input)
                user_input[UNIQUE_ID] = unique_id
                await self.async_set_unique_id(unique_id)
                self._abort_if_unique_id_configured()
                hostname = urlparse(user_input[CONF_URL]).hostname
                return self.async_create_entry(
                    title=f"OpenAI Compatible STT ({hostname}, {user_input[CONF_MODEL]})",
                    data=user_input,
                )
            except data_entry_flow.AbortFlow:
                return self.async_abort(reason="already_configured")
            except HomeAssistantError as e:
                _LOGGER.exception(str(e))
                errors["base"] = str(e)
            except ValueError as e:
                _LOGGER.exception(str(e))
                errors["base"] = str(e)
            except Exception as e:  # pylint: disable=broad-except
                _LOGGER.exception(str(e))
                errors["base"] = "unknown_error"
        return self.async_show_form(
            step_id="user",
            data_schema=self.data_schema,
            errors=errors,
            description_placeholders=user_input,
        )

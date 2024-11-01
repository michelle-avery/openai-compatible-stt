"""Setting up STT entity."""

from collections.abc import AsyncIterable
import logging
import os
import tempfile
import wave

from openai import AsyncOpenAI

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import generate_entity_id
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.httpx_client import get_async_client

from .const import (
    CONF_API_KEY,
    CONF_MODEL,
    CONF_URL,
    DOMAIN,
    SUPPORTED_LANGUAGES,
    UNIQUE_ID,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up OpenAI Compatible Speech-to-text platform via config entry."""

    api_key = None
    if CONF_API_KEY in config_entry.data:
        api_key = config_entry.data[CONF_API_KEY]

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=config_entry.data[CONF_URL],
        http_client=get_async_client(hass),
    )
    model = config_entry.data[CONF_MODEL]
    async_add_entities([OpenAICompatibleSTTEntity(hass, config_entry, client, model)])


class OpenAICompatibleSTTEntity(SpeechToTextEntity):
    """The OpenAI Compatible STT entity."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, hass, config, client, model):
        """Initialize STT entity."""
        self.hass = hass
        self._client = client
        self._config = config
        self._model = model

        self._attr_unique_id = config.data.get(UNIQUE_ID)
        if self._attr_unique_id is None:
            # generate a legacy unique_id
            self._attr_unique_id = f"{config.data[CONF_MODEL]}"
        self.entity_id = generate_entity_id(
            "stt.openai_compatible_stt_{}", config.data[CONF_MODEL], hass=hass
        )

    @property
    def default_language(self):
        """Return the default language."""
        return "en"

    @property
    def supported_languages(self):
        """Return the list of supported languages."""
        return SUPPORTED_LANGUAGES

    @property
    def device_info(self):
        return {
            "identifiers": {(DOMAIN, self._attr_unique_id)},
            "model": f"{self._config.data[CONF_MODEL]}",
            "manufacturer": "OpenAI Compatible",
        }

    @property
    def supported_formats(self):
        """Return a list of supported audio formats."""
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self):
        """Return a list of supported audio codecs."""
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self):
        """Return a list of supported audio bit rates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self):
        """Return a list of supported audio sample rates."""
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self):
        """Return a list of supported audio channels."""
        return [AudioChannels.CHANNEL_MONO]

    @property
    def name(self):
        """Return name of entity."""
        return f"{self._config.data[CONF_MODEL]}"

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Process the speech."""
        data = b""
        async for chunk in stream:
            data += chunk

        if not data:
            return SpeechResult("", SpeechResultState.ERROR)

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                with wave.open(temp_file, "wb") as wav_file:
                    wav_file.setnchannels(metadata.channel)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(metadata.sample_rate)
                    wav_file.writeframes(data)
                temp_file_path = temp_file.name

                transcription = await self._client.audio.transcriptions.create(
                    model=self._model, file=temp_file.file, language=metadata.language
                )
            return SpeechResult(transcription.text, SpeechResultState.SUCCESS)
        except Exception as e:
            return SpeechResult("", SpeechResultState.ERROR)

        # finally:
        #     if temp_file_path:
        #         os.remove(temp_file_path)

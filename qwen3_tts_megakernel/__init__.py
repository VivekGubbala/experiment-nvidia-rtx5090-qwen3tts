import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from mk_talker import install_megakernel_talker, MegakernelTalkerEngine  # noqa: E402
from mk_code_predictor import (  # noqa: E402
    install_megakernel_code_predictor, MegakernelCodePredictorEngine)
from mk_tts import StreamingMegakernelTTS, SR  # noqa: E402
from pc_tts import (  # noqa: E402
    MegakernelQwen3TTSService, MegakernelQwen3TTSWebsocketService)

__all__ = ["install_megakernel_talker", "MegakernelTalkerEngine",
           "install_megakernel_code_predictor", "MegakernelCodePredictorEngine",
           "StreamingMegakernelTTS", "SR", "MegakernelQwen3TTSService",
           "MegakernelQwen3TTSWebsocketService"]

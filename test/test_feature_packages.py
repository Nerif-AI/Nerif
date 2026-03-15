"""Tests for optional feature subpackage organization (nerif[img-gen], nerif[asr], nerif[tts])."""


# ---------------------------------------------------------------------------
# nerif.img_gen subpackage exports
# ---------------------------------------------------------------------------
class TestImgGenExports:
    def test_exports_image_generation_model(self):
        from nerif.img_gen import ImageGenerationModel

        assert ImageGenerationModel is not None

    def test_exports_nano_banana_model(self):
        from nerif.img_gen import NanoBananaModel

        assert NanoBananaModel is not None

    def test_exports_generated_image(self):
        from nerif.img_gen import GeneratedImage

        assert GeneratedImage is not None

    def test_exports_image_generation_result(self):
        from nerif.img_gen import ImageGenerationResult

        assert ImageGenerationResult is not None

    def test_all_list_matches_exports(self):
        import nerif.img_gen as pkg

        expected = {"GeneratedImage", "ImageGenerationModel", "ImageGenerationResult", "NanoBananaModel"}
        assert set(pkg.__all__) == expected


# ---------------------------------------------------------------------------
# nerif.asr subpackage exports
# ---------------------------------------------------------------------------
class TestAsrExports:
    def test_exports_audio_inference_model(self):
        from nerif.asr import AudioInferenceModel

        assert AudioInferenceModel is not None

    def test_exports_audio_model(self):
        from nerif.asr import AudioModel

        assert AudioModel is not None

    def test_exports_transcriber(self):
        from nerif.asr import Transcriber

        assert Transcriber is not None

    def test_all_list_matches_exports(self):
        import nerif.asr as pkg

        expected = {"AudioInferenceModel", "AudioModel", "Transcriber"}
        assert set(pkg.__all__) == expected


# ---------------------------------------------------------------------------
# nerif.tts subpackage exports
# ---------------------------------------------------------------------------
class TestTtsExports:
    def test_exports_speech_model(self):
        from nerif.tts import SpeechModel

        assert SpeechModel is not None

    def test_exports_audio_inference_model(self):
        from nerif.tts import AudioInferenceModel

        assert AudioInferenceModel is not None

    def test_exports_synthesizer(self):
        from nerif.tts import Synthesizer

        assert Synthesizer is not None

    def test_all_list_matches_exports(self):
        import nerif.tts as pkg

        expected = {"AudioInferenceModel", "SpeechModel", "Synthesizer"}
        assert set(pkg.__all__) == expected


# ---------------------------------------------------------------------------
# Lazy import via top-level `nerif` package
# ---------------------------------------------------------------------------
class TestLazyImport:
    def test_nerif_lazy_loads_img_gen(self):
        import nerif

        mod = nerif.img_gen
        assert hasattr(mod, "ImageGenerationModel")
        assert hasattr(mod, "NanoBananaModel")

    def test_nerif_lazy_loads_asr(self):
        import nerif

        mod = nerif.asr
        assert hasattr(mod, "AudioInferenceModel")
        assert hasattr(mod, "AudioModel")
        assert hasattr(mod, "Transcriber")

    def test_nerif_lazy_loads_tts(self):
        import nerif

        mod = nerif.tts
        assert hasattr(mod, "SpeechModel")
        assert hasattr(mod, "Synthesizer")

    def test_nerif_getattr_raises_for_unknown(self):
        import nerif

        try:
            _ = nerif.nonexistent_subpackage_xyz
            assert False, "Should have raised AttributeError"
        except AttributeError:
            pass

    def test_lazy_import_caches_module(self):
        """After first access the module should be cached in globals."""
        import nerif

        mod1 = nerif.img_gen
        mod2 = nerif.img_gen
        assert mod1 is mod2


# ---------------------------------------------------------------------------
# Core package should NOT export feature-specific classes
# ---------------------------------------------------------------------------
class TestModelPackageIsolation:
    def test_model_package_does_not_export_image_generation(self):
        import nerif.model

        assert "ImageGenerationModel" not in nerif.model.__all__
        assert "NanoBananaModel" not in nerif.model.__all__
        assert "GeneratedImage" not in nerif.model.__all__

    def test_model_package_does_not_export_audio_model(self):
        import nerif.model

        assert "AudioModel" not in nerif.model.__all__
        assert "SpeechModel" not in nerif.model.__all__
        assert "AudioInferenceModel" not in nerif.model.__all__

    def test_model_package_still_exports_core_classes(self):
        import nerif.model

        assert "SimpleChatModel" in nerif.model.__all__
        assert "LogitsChatModel" in nerif.model.__all__
        assert "MultiModalMessage" in nerif.model.__all__
        assert "VisionModel" in nerif.model.__all__
        assert "VideoModel" in nerif.model.__all__

    def test_top_level_init_does_not_eagerly_import_feature_packages(self):
        """Verify asr/tts/img_gen are not in nerif.__all__ (lazy only)."""
        import nerif

        assert "asr" not in nerif.__all__
        assert "tts" not in nerif.__all__
        assert "img_gen" not in nerif.__all__


# ---------------------------------------------------------------------------
# Cross-package class identity
# ---------------------------------------------------------------------------
class TestCrossPackageIdentity:
    def test_asr_and_tts_share_audio_inference_model_class(self):
        from nerif.asr import AudioInferenceModel as AsrAIM
        from nerif.tts import AudioInferenceModel as TtsAIM

        assert AsrAIM is TtsAIM
